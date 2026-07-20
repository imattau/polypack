import type { SerializedNode, SerializedEdge } from '../types.js'
import type { PersistenceAdapter, PersistenceChanges, PersistedNodeQuery } from './adapter.js'
import { applyPersistedNodeQuery } from './query.js'
import type { WalEntry } from './binary-format.js'
import { encodeWalEntries, decodeWalEntries, encodeSnapshot, decodeSnapshot } from './binary-format.js'
import type { FileIO } from './binary-file-io.js'
import { createFileIO } from './binary-file-io.js'

const SNAPSHOT_FILE = 'snapshot.msgpack'
const WAL_FILE = 'wal.msgpack'
const DEFAULT_COMPACT_THRESHOLD = 10_000

export interface BinaryStoreConfig {
  storeDir: string
  compactThreshold?: number
  fileIO?: FileIO
}

interface ResolvedBinaryStoreConfig {
  storeDir: string
  compactThreshold: number
  fileIO?: FileIO
}

export class BinaryStoreAdapter implements PersistenceAdapter {
  private nodes = new Map<string, SerializedNode>()
  private edges = new Map<string, SerializedEdge>()
  private vectors = new Map<string, number[]>()
  private io!: FileIO
  private config: ResolvedBinaryStoreConfig
  private loaded = false
  private walEntryCount = 0
  private compactTimer: ReturnType<typeof setTimeout> | null = null
  private compacting = false

  constructor(config: BinaryStoreConfig) {
    this.config = {
      storeDir: config.storeDir,
      compactThreshold: config.compactThreshold ?? DEFAULT_COMPACT_THRESHOLD,
      fileIO: config.fileIO,
    }
  }

  private async ensureLoaded(): Promise<void> {
    if (this.loaded) return
    this.loaded = true
    this.io = this.config.fileIO ?? (await createFileIO(this.config.storeDir))
    const snapshotData = await this.io.readFile(SNAPSHOT_FILE)
    if (snapshotData) {
      const { nodes, edges, vectors } = decodeSnapshot(snapshotData)
      this.nodes = nodes
      this.edges = edges
      this.vectors = vectors
    }
    const walData = await this.io.readFile(WAL_FILE)
    if (walData && walData.length > 0) {
      for (const entry of decodeWalEntries(walData)) {
        this.replayEntry(entry)
      }
      this.walEntryCount = 0
      await this.io.deleteFile(WAL_FILE)
      await this.writeSnapshot()
    }
  }

  private replayEntry(entry: WalEntry): void {
    switch (entry.kind) {
      case 'putNode':
        this.nodes.set(entry.node.id, entry.node)
        break
      case 'deleteNode':
        this.nodes.delete(entry.id)
        break
      case 'putEdge':
        this.edges.set(entry.edge.id, entry.edge)
        break
      case 'deleteEdge':
        this.edges.delete(entry.id)
        break
      case 'putVector':
        this.vectors.set(entry.id, entry.vector)
        break
      case 'deleteVector':
        this.vectors.delete(entry.id)
        break
      case 'clearAll':
        this.nodes.clear()
        this.edges.clear()
        this.vectors.clear()
        break
    }
  }

  private async flushWAL(): Promise<void> {
    if (this.walEntryCount === 0) return
    await this.writeSnapshot()
    await this.io.writeFile(WAL_FILE, new Uint8Array(0))
    this.walEntryCount = 0
  }

  private async writeSnapshot(): Promise<void> {
    await this.io.writeFile(SNAPSHOT_FILE, encodeSnapshot(this.nodes, this.edges, this.vectors))
  }

  private scheduleCompact(): void {
    if (this.compactTimer || this.compacting) return
    this.compactTimer = setTimeout(() => {
      this.compactTimer = null
      this.compact().catch(err => console.warn('[BinaryStoreAdapter] compact error:', err))
    }, 100)
  }

  private async compact(): Promise<void> {
    if (this.compacting) return
    this.compacting = true
    try {
      await this.flushWAL()
    } finally {
      this.compacting = false
    }
  }

  async applyChanges(changes: PersistenceChanges): Promise<void> {
    await this.ensureLoaded()
    const entries: WalEntry[] = []
    for (const id of changes.deleteNodeIds) {
      this.nodes.delete(id)
      entries.push({ kind: 'deleteNode', id })
    }
    for (const id of changes.deleteEdgeIds) {
      this.edges.delete(id)
      entries.push({ kind: 'deleteEdge', id })
    }
    for (const id of changes.deleteVectorIds) {
      this.vectors.delete(id)
      entries.push({ kind: 'deleteVector', id })
    }
    for (const node of changes.putNodes) {
      this.nodes.set(node.id, node)
      entries.push({ kind: 'putNode', node })
    }
    for (const edge of changes.putEdges) {
      this.edges.set(edge.id, edge)
      entries.push({ kind: 'putEdge', edge })
    }
    for (const entry of changes.putVectors) {
      this.vectors.set(entry.id, entry.vector)
      entries.push({ kind: 'putVector', id: entry.id, vector: entry.vector })
    }
    if (entries.length > 0) {
      const encoded = encodeWalEntries(entries)
      await this.io.appendFile(WAL_FILE, encoded)
      this.walEntryCount += entries.length
      if (this.walEntryCount >= this.config.compactThreshold) {
        this.scheduleCompact()
      }
    }
  }

  async putNode(node: SerializedNode): Promise<void> {
    await this.applyChanges({ putNodes: [node], deleteNodeIds: [], putEdges: [], deleteEdgeIds: [], putVectors: [], deleteVectorIds: [] })
  }

  async bulkPutNodes(nodes: SerializedNode[]): Promise<void> {
    if (nodes.length === 0) return
    await this.applyChanges({ putNodes: nodes, deleteNodeIds: [], putEdges: [], deleteEdgeIds: [], putVectors: [], deleteVectorIds: [] })
  }

  async getNode(id: string): Promise<SerializedNode | undefined> {
    await this.ensureLoaded()
    return this.nodes.get(id)
  }

  async getNodes(ids: string[]): Promise<SerializedNode[]> {
    await this.ensureLoaded()
    return ids.map(id => this.nodes.get(id)).filter(Boolean) as SerializedNode[]
  }

  async deleteNode(id: string): Promise<void> {
    await this.applyChanges({ putNodes: [], deleteNodeIds: [id], putEdges: [], deleteEdgeIds: [], putVectors: [], deleteVectorIds: [] })
  }

  async bulkDeleteNodes(ids: string[]): Promise<void> {
    if (ids.length === 0) return
    await this.applyChanges({ putNodes: [], deleteNodeIds: ids, putEdges: [], deleteEdgeIds: [], putVectors: [], deleteVectorIds: [] })
  }

  async allNodeIds(): Promise<string[]> {
    await this.ensureLoaded()
    return [...this.nodes.keys()]
  }

  async queryNodes(query: PersistedNodeQuery): Promise<SerializedNode[]> {
    await this.ensureLoaded()
    return applyPersistedNodeQuery([...this.nodes.values()], query)
  }

  async countNodes(query: PersistedNodeQuery): Promise<number> {
    await this.ensureLoaded()
    const all = [...this.nodes.values()]
    const filtered = applyPersistedNodeQuery(all, query)
    return filtered.length
  }

  async putEdge(edge: SerializedEdge): Promise<void> {
    await this.applyChanges({ putNodes: [], deleteNodeIds: [], putEdges: [edge], deleteEdgeIds: [], putVectors: [], deleteVectorIds: [] })
  }

  async bulkPutEdges(edges: SerializedEdge[]): Promise<void> {
    if (edges.length === 0) return
    await this.applyChanges({ putNodes: [], deleteNodeIds: [], putEdges: edges, deleteEdgeIds: [], putVectors: [], deleteVectorIds: [] })
  }

  async getAllEdges(): Promise<SerializedEdge[]> {
    await this.ensureLoaded()
    return [...this.edges.values()]
  }

  async getEdgesBySources(sources: string[], type?: string): Promise<SerializedEdge[]> {
    await this.ensureLoaded()
    const sourceSet = new Set(sources)
    return [...this.edges.values()].filter(e => sourceSet.has(e.source) && (type === undefined || e.type === type))
  }

  async getEdgesByTargets(targets: string[], type?: string): Promise<SerializedEdge[]> {
    await this.ensureLoaded()
    const targetSet = new Set(targets)
    return [...this.edges.values()].filter(e => targetSet.has(e.target) && (type === undefined || e.type === type))
  }

  async deleteEdge(id: string): Promise<void> {
    await this.applyChanges({ putNodes: [], deleteNodeIds: [], putEdges: [], deleteEdgeIds: [id], putVectors: [], deleteVectorIds: [] })
  }

  async bulkDeleteEdges(ids: string[]): Promise<void> {
    if (ids.length === 0) return
    await this.applyChanges({ putNodes: [], deleteNodeIds: [], putEdges: [], deleteEdgeIds: ids, putVectors: [], deleteVectorIds: [] })
  }

  async putVector(id: string, vector: number[]): Promise<void> {
    await this.applyChanges({ putNodes: [], deleteNodeIds: [], putEdges: [], deleteEdgeIds: [], putVectors: [{ id, vector }], deleteVectorIds: [] })
  }

  async bulkPutVectors(entries: Array<{ id: string; vector: number[] }>): Promise<void> {
    if (entries.length === 0) return
    await this.applyChanges({ putNodes: [], deleteNodeIds: [], putEdges: [], deleteEdgeIds: [], putVectors: entries, deleteVectorIds: [] })
  }

  async deleteVector(id: string): Promise<void> {
    await this.applyChanges({ putNodes: [], deleteNodeIds: [], putEdges: [], deleteEdgeIds: [], putVectors: [], deleteVectorIds: [id] })
  }

  async getVectors(ids: string[]): Promise<Array<{ id: string; vector: number[] }>> {
    await this.ensureLoaded()
    const results: Array<{ id: string; vector: number[] }> = []
    for (const id of ids) {
      const vector = this.vectors.get(id)
      if (vector) results.push({ id, vector })
    }
    return results
  }

  async getAllVectors(): Promise<Array<{ id: string; vector: number[] }>> {
    await this.ensureLoaded()
    return [...this.vectors.entries()].map(([id, vector]) => ({ id, vector }))
  }

  async clearAll(): Promise<void> {
    await this.ensureLoaded()
    this.nodes.clear()
    this.edges.clear()
    this.vectors.clear()
    this.walEntryCount = 0
    await this.io.writeFile(SNAPSHOT_FILE, encodeSnapshot(this.nodes, this.edges, this.vectors))
    await this.io.writeFile(WAL_FILE, new Uint8Array(0))
  }

  async close(): Promise<void> {
    if (this.compactTimer) {
      clearTimeout(this.compactTimer)
      this.compactTimer = null
    }
    if (this.loaded) {
      await this.compact()
    }
  }
}

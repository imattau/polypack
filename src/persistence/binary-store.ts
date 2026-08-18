import type { SerializedNode, SerializedEdge, AdapterCapabilities, IndexDefinition, MutationRecord, VerificationReport } from '../types.js'
import type { PersistenceAdapter, PersistenceChanges, PersistedNodeQuery } from './adapter.js'
import { applyPersistedCountPagination, applyPersistedNodeQuery, matchesPersistedNode } from './query.js'
import type { WalEntry } from './binary-format.js'
import { encodeWalEntries, decodeWalEntries, encodeSnapshot, decodeSnapshot, encodeMutationRecords, decodeMutationRecords } from './binary-format.js'
import type { FileIO } from './file-io.js'
import { createFileIO } from './file-io.js'
import { mutationRecordFromChanges } from './mutation-log.js'
import { ReadOnlyStoreError } from './lock-errors.js'

const SNAPSHOT_FILE = 'snapshot.msgpack'
const WAL_FILE = 'wal.msgpack'
const MUTATION_LOG_FILE = 'mutations.msgpack'
const LOCK_FILE = 'store.lock'
const DEFAULT_STALE_LOCK_MS = 30_000
const DEFAULT_COMPACT_THRESHOLD = 10_000
/** Compact once the WAL holds at least this share of the store's record count. */
const COMPACT_RATIO = 4

export interface BinaryStoreConfig {
  /** Directory holding `snapshot.msgpack` and `wal.msgpack`. */
  storeDir: string
  /**
   * Minimum WAL-entry count at which compaction is scheduled (default
   * 10,000). Acts as a lower bound — the effective threshold also grows with
   * the store (`max(compactThreshold, records / 4)`) so a large store
   * doesn't rewrite its snapshot on every batch.
   */
  compactThreshold?: number
  /** Byte-storage backend. Defaults to a platform-appropriate implementation created at first use. */
  fileIO?: FileIO
  /** fsync WAL appends and snapshot writes. Off by default; see durability docs. */
  syncWrites?: boolean
  /** Open without a writer lock and reject mutations. */
  readOnly?: boolean
  /** Age after which an abandoned writer lock may be recovered. */
  staleLockMs?: number
}

interface ResolvedBinaryStoreConfig {
  storeDir: string
  compactThreshold: number
  fileIO?: FileIO
  syncWrites: boolean
  readOnly: boolean
  staleLockMs: number
}

/**
 * Persists nodes, edges, and vectors as a MessagePack snapshot plus an
 * append-only write-ahead log. Nodes, edges, and vectors are committed
 * atomically per batch via {@link applyChanges}; the WAL is compacted into a
 * fresh snapshot once it passes an adaptive threshold and on {@link close}.
 * Startup replays the WAL, then writes a snapshot before deleting it, so a
 * crash between those two steps loses nothing; a truncated WAL tail from a
 * mid-append crash is also tolerated.
 */
export class BinaryStoreAdapter implements PersistenceAdapter {
  readonly capabilities: AdapterCapabilities
  private nodes = new Map<string, SerializedNode>()
  private edges = new Map<string, SerializedEdge>()
  private vectors = new Map<string, number[]>()
  private indexDefinitions: IndexDefinition[] = []
  private mutationRecords: MutationRecord[] = []
  private nextMutationSequence = 1n
  private byType = new Map<string, Set<string>>()
  private edgesBySource = new Map<string, Set<string>>()
  private edgesByTarget = new Map<string, Set<string>>()
  private io!: FileIO
  private config: ResolvedBinaryStoreConfig
  private walEntryCount = 0
  private compactTimer: ReturnType<typeof setTimeout> | null = null
  private initPromise: Promise<void> | null = null
  private closed = false
  private queue: Promise<unknown> = Promise.resolve()
  private releaseLock?: () => Promise<void>

  constructor(config: BinaryStoreConfig) {
    this.config = {
      storeDir: config.storeDir,
      compactThreshold: config.compactThreshold ?? DEFAULT_COMPACT_THRESHOLD,
      fileIO: config.fileIO,
      syncWrites: config.syncWrites ?? false,
      readOnly: config.readOnly ?? false,
      staleLockMs: config.staleLockMs ?? DEFAULT_STALE_LOCK_MS,
    }
    this.capabilities = {
      atomicBatches: true,
      transactions: true,
      fsync: this.config.syncWrites,
      secondaryIndexes: true,
      snapshots: true,
      changeFeed: false,
      concurrentWriters: false,
      vectorSearch: 'exact',
    }
  }

  /** Serialise initialisation, mutation, compaction, and shutdown. */
  private enqueue<T>(fn: () => Promise<T>): Promise<T> {
    const run = this.queue.then(fn)
    this.queue = run.then(() => undefined, () => undefined)
    return run
  }

  // ── secondary indexes ──

  private indexNode(node: SerializedNode): void {
    const existing = this.nodes.get(node.id)
    if (existing && existing.type !== node.type) this.unindexNode(node.id, existing.type)
    let ids = this.byType.get(node.type)
    if (!ids) {
      ids = new Set()
      this.byType.set(node.type, ids)
    }
    ids.add(node.id)
  }

  private unindexNode(id: string, type?: string): void {
    const node = this.nodes.get(id)
    const ids = this.byType.get(type ?? node?.type ?? '')
    if (ids) {
      ids.delete(id)
      if (ids.size === 0) this.byType.delete(type ?? node?.type ?? '')
    }
  }

  private indexEdge(edge: SerializedEdge): void {
    let bySource = this.edgesBySource.get(edge.source)
    if (!bySource) {
      bySource = new Set()
      this.edgesBySource.set(edge.source, bySource)
    }
    bySource.add(edge.id)
    let byTarget = this.edgesByTarget.get(edge.target)
    if (!byTarget) {
      byTarget = new Set()
      this.edgesByTarget.set(edge.target, byTarget)
    }
    byTarget.add(edge.id)
  }

  private unindexEdge(id: string): void {
    const edge = this.edges.get(id)
    if (!edge) return
    const bySource = this.edgesBySource.get(edge.source)
    if (bySource) {
      bySource.delete(id)
      if (bySource.size === 0) this.edgesBySource.delete(edge.source)
    }
    const byTarget = this.edgesByTarget.get(edge.target)
    if (byTarget) {
      byTarget.delete(id)
      if (byTarget.size === 0) this.edgesByTarget.delete(edge.target)
    }
  }

  /** Rebuild the secondary indexes from the full record maps. */
  private rebuildIndexes(): void {
    this.byType.clear()
    this.edgesBySource.clear()
    this.edgesByTarget.clear()
    for (const node of this.nodes.values()) this.indexNode(node)
    for (const edge of this.edges.values()) this.indexEdge(edge)
  }

  /** Effective WAL compaction threshold, grown with the store to keep total
   *  compaction work linear in writes instead of quadratic. */
  private effectiveCompactThreshold(): number {
    const records = this.nodes.size + this.edges.size + this.vectors.size
    return Math.max(this.config.compactThreshold, Math.floor(records / COMPACT_RATIO))
  }

  /** Candidate node ids for a query when only the type index is needed. */
  private typeCandidateIds(query: PersistedNodeQuery): string[] | null {
    if (!query.nodeTypes || query.nodeTypes.length === 0) return null
    if (query.attributes || query.attributeRanges) return null
    const ids = new Set<string>()
    for (const type of query.nodeTypes) {
      const set = this.byType.get(type)
      if (set) for (const id of set) ids.add(id)
    }
    return [...ids]
  }

  private ensureLoaded(): Promise<void> {
    if (!this.initPromise) {
      this.initPromise = this.doLoad().catch((err) => {
        this.initPromise = null
        throw err
      })
    }
    return this.initPromise
  }

  private assertOpen(): void {
    if (this.closed) throw new Error('BinaryStoreAdapter is closed')
  }

  private assertWritable(): void {
    if (this.config.readOnly) throw new ReadOnlyStoreError()
  }

  private async doLoad(): Promise<void> {
    this.io = this.config.fileIO ?? (await createFileIO(this.config.storeDir, this.config.syncWrites))
    if (!this.config.readOnly && this.io.acquireExclusiveLock) {
      this.releaseLock = await this.io.acquireExclusiveLock(LOCK_FILE, { storeDir: this.config.storeDir }, this.config.staleLockMs)
    }
    const snapshotData = await this.io.readFile(SNAPSHOT_FILE)
    if (snapshotData) {
      const { nodes, edges, vectors, indexes } = decodeSnapshot(snapshotData)
      this.nodes = nodes
      this.edges = edges
      this.vectors = vectors
      this.indexDefinitions = indexes
    }
    const walData = await this.io.readFile(WAL_FILE)
    if (walData && walData.length > 0) {
      // Replay the WAL, then persist a snapshot BEFORE deleting the WAL. A crash
      // between the snapshot write and WAL deletion only re-replays an already
      // applied (idempotent) WAL; the old delete-first ordering could lose the
      // recovered changes on a crash.
      for (const entry of decodeWalEntries(walData)) {
        this.replayEntry(entry)
      }
      if (!this.config.readOnly) {
        await this.writeSnapshot()
        await this.io.deleteFile(WAL_FILE)
        this.walEntryCount = 0
      }
    }
    this.rebuildIndexes()
    const mutationData = await this.io.readFile(MUTATION_LOG_FILE)
    if (mutationData && mutationData.length > 0) {
      this.mutationRecords = [...decodeMutationRecords(mutationData)]
      if (this.mutationRecords.length > 0) {
        this.nextMutationSequence = this.mutationRecords[this.mutationRecords.length - 1].sequence + 1n
      }
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
      case 'setIndexes':
        this.indexDefinitions = entry.indexes.map(index => ({ ...index, fields: [...index.fields] }))
        break
    }
  }

  private async writeSnapshot(): Promise<void> {
    await this.io.writeFile(SNAPSHOT_FILE, encodeSnapshot(this.nodes, this.edges, this.vectors, this.indexDefinitions))
  }

  /**
   * Compact the WAL into a snapshot. Runs under the operation queue. The
   * generation captured at entry is the number of WAL entries reflected in the
   * snapshot; if (defensively) more entries appear mid-write, only those are
   * truncated and the remainder is preserved in the WAL.
   */
  private async compact(): Promise<void> {
    const generation = this.walEntryCount
    if (generation === 0) return
    await this.writeSnapshot()
    const walData = await this.io.readFile(WAL_FILE)
    let remaining = 0
    if (walData && walData.length > 0) {
      const entries = [...decodeWalEntries(walData)]
      remaining = entries.length - generation
      if (remaining > 0) {
        await this.io.writeFile(WAL_FILE, encodeWalEntries(entries.slice(generation)))
      } else {
        await this.io.writeFile(WAL_FILE, new Uint8Array(0))
      }
    }
    this.walEntryCount = remaining
  }

  private scheduleCompact(): void {
    if (this.compactTimer) return
    this.compactTimer = setTimeout(() => {
      this.compactTimer = null
      this.enqueue(() => this.compact()).catch((err) => console.warn('[BinaryStoreAdapter] compact error:', err))
    }, 100)
  }

  async applyChanges(changes: PersistenceChanges): Promise<void> {
    this.assertOpen()
    this.assertWritable()
    await this.enqueue(async () => {
      await this.ensureLoaded()
      if (changes.operationId && this.mutationRecords.some(record => record.operationId === changes.operationId)) return
      const entries: WalEntry[] = []
      if (changes.indexDefinitions) {
        this.indexDefinitions = changes.indexDefinitions.map(index => ({ ...index, fields: [...index.fields] }))
        entries.push({ kind: 'setIndexes', indexes: this.indexDefinitions })
      }
      for (const id of changes.deleteNodeIds) {
        const node = this.nodes.get(id)
        this.unindexNode(id, node?.type)
        this.nodes.delete(id)
        entries.push({ kind: 'deleteNode', id })
      }
      for (const id of changes.deleteEdgeIds) {
        this.unindexEdge(id)
        this.edges.delete(id)
        entries.push({ kind: 'deleteEdge', id })
      }
      for (const id of changes.deleteVectorIds) {
        this.vectors.delete(id)
        entries.push({ kind: 'deleteVector', id })
      }
      for (const node of changes.putNodes) {
        this.indexNode(node)
        this.nodes.set(node.id, node)
        entries.push({ kind: 'putNode', node })
      }
      for (const edge of changes.putEdges) {
        this.edges.set(edge.id, edge)
        this.indexEdge(edge)
        entries.push({ kind: 'putEdge', edge })
      }
      for (const entry of changes.putVectors) {
        this.vectors.set(entry.id, entry.vector)
        entries.push({ kind: 'putVector', id: entry.id, vector: entry.vector })
      }
      if (entries.length > 0) {
        const encoded = encodeWalEntries(entries)
        await this.io.appendFile(WAL_FILE, encoded)
        const record = mutationRecordFromChanges(changes, this.nextMutationSequence)
        if (record) {
          await this.io.appendFile(MUTATION_LOG_FILE, encodeMutationRecords([record]))
          this.mutationRecords.push(record)
          this.nextMutationSequence++
        }
        this.walEntryCount += entries.length
        if (this.walEntryCount >= this.effectiveCompactThreshold()) {
          this.scheduleCompact()
        }
      }
    })
  }

  async getIndexDefinitions(): Promise<IndexDefinition[]> {
    this.assertOpen()
    await this.enqueue(() => this.ensureLoaded())
    return this.indexDefinitions.map(index => ({ ...index, fields: [...index.fields] }))
  }

  /** Force the current state into a snapshot and compact the recovery WAL. */
  async checkpoint(): Promise<void> {
    this.assertOpen()
    this.assertWritable()
    await this.enqueue(async () => {
      await this.ensureLoaded()
      await this.compact()
      if (this.walEntryCount === 0) await this.writeSnapshot()
    })
  }

  /** Verify decoded records, indexes, vectors, and logical mutation history. */
  async verify(): Promise<VerificationReport> {
    this.assertOpen()
    await this.enqueue(() => this.ensureLoaded())
    const errors: string[] = []
    for (const node of this.nodes.values()) {
      if (!node.id || !node.type) errors.push(`invalid node identity: ${node.id}`)
      if (!Number.isFinite(node.insertedAt) || !Number.isFinite(node.updatedAt)) errors.push(`invalid timestamps: ${node.id}`)
      if (node.activation) {
        if (!Number.isFinite(node.activation.score) || node.activation.score < 0 || node.activation.score > 1) errors.push(`invalid activation score: ${node.id}`)
        if (!Number.isFinite(node.activation.importance) || node.activation.importance < 0 || node.activation.importance > 1) errors.push(`invalid activation importance: ${node.id}`)
        if (!Number.isInteger(node.activation.reinforcementCount) || node.activation.reinforcementCount < 0) errors.push(`invalid activation count: ${node.id}`)
        if (!Number.isFinite(node.activation.lastMeaningfulActivation) || node.activation.lastMeaningfulActivation < 0) errors.push(`invalid activation timestamp: ${node.id}`)
      }
      const storedVector = this.vectors.get(node.id)
      if (node.vector && storedVector && node.vector.length !== storedVector.length) errors.push(`vector dimensionality mismatch: ${node.id}`)
    }
    for (const edge of this.edges.values()) {
      if (!edge.id || !edge.source || !edge.target || !edge.type) errors.push(`invalid edge identity: ${edge.id}`)
      if (!this.nodes.has(edge.source)) errors.push(`edge ${edge.id} has missing source ${edge.source}`)
      if (!this.nodes.has(edge.target)) errors.push(`edge ${edge.id} has missing target ${edge.target}`)
    }
    for (const [id, vector] of this.vectors) {
      if (!vector.every(value => Number.isFinite(value))) errors.push(`invalid vector values: ${id}`)
    }
    for (const [type, ids] of this.byType) {
      for (const id of ids) if (this.nodes.get(id)?.type !== type) errors.push(`stale type index entry: ${type}/${id}`)
    }
    for (const [id, edgeIds] of this.edgesBySource) {
      for (const edgeId of edgeIds) if (this.edges.get(edgeId)?.source !== id) errors.push(`stale source edge index entry: ${id}/${edgeId}`)
    }
    for (const [id, edgeIds] of this.edgesByTarget) {
      for (const edgeId of edgeIds) if (this.edges.get(edgeId)?.target !== id) errors.push(`stale target edge index entry: ${id}/${edgeId}`)
    }
    for (let i = 1; i < this.mutationRecords.length; i++) {
      if (this.mutationRecords[i].sequence <= this.mutationRecords[i - 1].sequence) errors.push(`non-monotonic mutation sequence at index ${i}`)
    }
    return { ok: errors.length === 0, errors, nodeCount: this.nodes.size, edgeCount: this.edges.size, vectorCount: this.vectors.size, mutationCount: this.mutationRecords.length }
  }

  async stats() {
    this.assertOpen()
    await this.enqueue(() => this.ensureLoaded())
    const wal = await this.io.readFile(WAL_FILE)
    return {
      persistedNodeCount: this.nodes.size,
      edgeCount: this.edges.size,
      vectorCount: this.vectors.size,
      mutationCount: this.mutationRecords.length,
      walBytes: wal?.byteLength ?? 0,
    }
  }

  /** Copy a checkpointed store into another FileIO directory. */
  async backup(destination: FileIO): Promise<void> {
    if (this.config.readOnly) {
      this.assertOpen()
      await this.enqueue(() => this.ensureLoaded())
    } else {
      await this.checkpoint()
    }
    for (const name of [SNAPSHOT_FILE, WAL_FILE, MUTATION_LOG_FILE]) {
      const data = await this.io.readFile(name)
      if (data) await destination.writeFile(name, data)
    }
  }

  /** Restore all store files from one FileIO directory into another. */
  static async restore(source: FileIO, destination: FileIO): Promise<void> {
    for (const name of [SNAPSHOT_FILE, WAL_FILE, MUTATION_LOG_FILE]) {
      const data = await source.readFile(name)
      if (data) await destination.writeFile(name, data)
      else await destination.deleteFile(name)
    }
  }

  async getMutationsSince(sequence: bigint): Promise<MutationRecord[]> {
    this.assertOpen()
    await this.enqueue(() => this.ensureLoaded())
    return this.mutationRecords
      .filter(record => record.sequence > sequence)
      .map(record => ({ ...record, operations: record.operations.map(operation => ({ ...operation, payload: structuredClone(operation.payload) })) }))
  }

  async latestMutationSequence(): Promise<bigint> {
    this.assertOpen()
    await this.enqueue(() => this.ensureLoaded())
    return this.nextMutationSequence - 1n
  }

  async putNode(node: SerializedNode): Promise<void> {
    await this.applyChanges({ putNodes: [node], deleteNodeIds: [], putEdges: [], deleteEdgeIds: [], putVectors: [], deleteVectorIds: [] })
  }

  async bulkPutNodes(nodes: SerializedNode[]): Promise<void> {
    if (nodes.length === 0) return
    await this.applyChanges({ putNodes: nodes, deleteNodeIds: [], putEdges: [], deleteEdgeIds: [], putVectors: [], deleteVectorIds: [] })
  }

  async getNode(id: string): Promise<SerializedNode | undefined> {
    this.assertOpen()
    return this.enqueue(async () => {
      await this.ensureLoaded()
      return this.nodes.get(id)
    })
  }

  async getNodes(ids: string[]): Promise<SerializedNode[]> {
    this.assertOpen()
    return this.enqueue(async () => {
      await this.ensureLoaded()
      return ids.map(id => this.nodes.get(id)).filter(Boolean) as SerializedNode[]
    })
  }

  async deleteNode(id: string): Promise<void> {
    await this.applyChanges({ putNodes: [], deleteNodeIds: [id], putEdges: [], deleteEdgeIds: [], putVectors: [], deleteVectorIds: [] })
  }

  async bulkDeleteNodes(ids: string[]): Promise<void> {
    if (ids.length === 0) return
    await this.applyChanges({ putNodes: [], deleteNodeIds: ids, putEdges: [], deleteEdgeIds: [], putVectors: [], deleteVectorIds: [] })
  }

  async allNodeIds(): Promise<string[]> {
    this.assertOpen()
    return this.enqueue(async () => {
      await this.ensureLoaded()
      return [...this.nodes.keys()]
    })
  }

  async queryNodes(query: PersistedNodeQuery): Promise<SerializedNode[]> {
    this.assertOpen()
    return this.enqueue(async () => {
      await this.ensureLoaded()
      const candidates = this.typeCandidateIds(query)
      const nodes = candidates
        ? candidates.map(id => this.nodes.get(id)).filter((n): n is SerializedNode => !!n)
        : [...this.nodes.values()]
      return applyPersistedNodeQuery(nodes, query)
    })
  }

  async countNodes(query: PersistedNodeQuery): Promise<number> {
    this.assertOpen()
    return this.enqueue(async () => {
      await this.ensureLoaded()
      const isTypeOnly = !!query.nodeTypes && query.nodeTypes.length > 0 &&
        !query.attributes && !query.attributeRanges
      let count: number
      if (!query.nodeTypes && !query.attributes && !query.attributeRanges) {
        count = this.nodes.size
      } else if (isTypeOnly) {
        count = 0
        for (const type of query.nodeTypes!) {
          count += this.byType.get(type)?.size ?? 0
        }
      } else {
        count = 0
        for (const node of this.nodes.values()) {
          if (matchesPersistedNode(node, query)) count++
        }
      }
      return applyPersistedCountPagination(count, query)
    })
  }

  async putEdge(edge: SerializedEdge): Promise<void> {
    await this.applyChanges({ putNodes: [], deleteNodeIds: [], putEdges: [edge], deleteEdgeIds: [], putVectors: [], deleteVectorIds: [] })
  }

  async bulkPutEdges(edges: SerializedEdge[]): Promise<void> {
    if (edges.length === 0) return
    await this.applyChanges({ putNodes: [], deleteNodeIds: [], putEdges: edges, deleteEdgeIds: [], putVectors: [], deleteVectorIds: [] })
  }

  async getAllEdges(): Promise<SerializedEdge[]> {
    this.assertOpen()
    return this.enqueue(async () => {
      await this.ensureLoaded()
      return [...this.edges.values()]
    })
  }

  async getEdgesBySources(sources: string[], type?: string): Promise<SerializedEdge[]> {
    this.assertOpen()
    return this.enqueue(async () => {
      await this.ensureLoaded()
      const out: SerializedEdge[] = []
      const seen = new Set<string>()
      for (const source of sources) {
        const ids = this.edgesBySource.get(source)
        if (!ids) continue
        for (const id of ids) {
          const edge = this.edges.get(id)
          if (edge && (type === undefined || edge.type === type) && !seen.has(id)) {
            seen.add(id)
            out.push(edge)
          }
        }
      }
      return out
    })
  }

  async getEdgesByTargets(targets: string[], type?: string): Promise<SerializedEdge[]> {
    this.assertOpen()
    return this.enqueue(async () => {
      await this.ensureLoaded()
      const out: SerializedEdge[] = []
      const seen = new Set<string>()
      for (const target of targets) {
        const ids = this.edgesByTarget.get(target)
        if (!ids) continue
        for (const id of ids) {
          const edge = this.edges.get(id)
          if (edge && (type === undefined || edge.type === type) && !seen.has(id)) {
            seen.add(id)
            out.push(edge)
          }
        }
      }
      return out
    })
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
    this.assertOpen()
    return this.enqueue(async () => {
      await this.ensureLoaded()
      const results: Array<{ id: string; vector: number[] }> = []
      for (const id of ids) {
        const vector = this.vectors.get(id)
        if (vector) results.push({ id, vector })
      }
      return results
    })
  }

  async getAllVectors(): Promise<Array<{ id: string; vector: number[] }>> {
    this.assertOpen()
    return this.enqueue(async () => {
      await this.ensureLoaded()
      return [...this.vectors.entries()].map(([id, vector]) => ({ id, vector }))
    })
  }

  async clearAll(): Promise<void> {
    this.assertOpen()
    this.assertWritable()
    await this.enqueue(async () => {
      await this.ensureLoaded()
      this.nodes.clear()
      this.edges.clear()
      this.vectors.clear()
      this.byType.clear()
      this.edgesBySource.clear()
      this.edgesByTarget.clear()
      this.walEntryCount = 0
      await this.io.writeFile(SNAPSHOT_FILE, encodeSnapshot(this.nodes, this.edges, this.vectors, this.indexDefinitions))
      await this.io.writeFile(WAL_FILE, new Uint8Array(0))
    })
  }

  async close(): Promise<void> {
    if (this.compactTimer) {
      clearTimeout(this.compactTimer)
      this.compactTimer = null
    }
    // Flip synchronously, before enqueueing: a write whose assertOpen() check
    // races with close() must see the closed state immediately, not just once
    // this task reaches the front of the queue — otherwise it can slip in and
    // execute after the compaction step below.
    this.closed = true
    await this.enqueue(async () => {
      if (this.initPromise) {
        await this.ensureLoaded()
        if (!this.config.readOnly) await this.compact()
      }
      if (this.releaseLock) {
        await this.releaseLock()
        this.releaseLock = undefined
      }
    })
  }
}

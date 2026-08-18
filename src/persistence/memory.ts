import type { SerializedNode, SerializedEdge, IndexDefinition, MutationRecord } from '../types.js'
import type { PersistenceAdapter, PersistenceChanges, PersistedNodeQuery } from './adapter.js'
import { applyPersistedCountPagination, applyPersistedNodeQuery, matchesPersistedNode } from './query.js'
import { mutationRecordFromChanges } from './mutation-log.js'

/** Volatile persistence adapter for Node.js, tests, and temporary graphs. */
export class MemoryAdapter implements PersistenceAdapter {
  readonly capabilities = {
    atomicBatches: true,
    transactions: true,
    fsync: false,
    secondaryIndexes: true,
    snapshots: true,
    changeFeed: false,
    concurrentWriters: false,
    vectorSearch: 'exact' as const,
  }
  private nodes = new Map<string, SerializedNode>()
  private edges = new Map<string, SerializedEdge>()
  private vectors = new Map<string, number[]>()
  private byType = new Map<string, Set<string>>()
  private edgesBySource = new Map<string, Set<string>>()
  private edgesByTarget = new Map<string, Set<string>>()
  private nodeOrder = new Map<string, true>()
  private indexDefinitions: IndexDefinition[] = []
  private mutations: MutationRecord[] = []
  private nextMutation = 1n
  private readonly maxNodes: number | undefined

  /**
   * @param maxNodes When set, evicts the least-recently-put node (and its
   * edges/vector) once this many nodes are stored. Unbounded by default.
   * A node's position is bumped to most-recently-put whenever it's written
   * again, whether through `putNode`/`bulkPutNodes` or `applyChanges` (the
   * path `PolyGraph.flush()`/`save()` use).
   */
  constructor(maxNodes?: number) {
    this.maxNodes = maxNodes
  }

  private touchNode(id: string): void {
    this.nodeOrder.delete(id)
    this.nodeOrder.set(id, true)
  }

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

  private evictIfOverCap(): void {
    if (this.maxNodes === undefined) return
    while (this.nodeOrder.size > this.maxNodes) {
      const evict = this.nodeOrder.keys().next().value
      if (evict === undefined) break
      this.unindexNode(evict)
      this.nodeOrder.delete(evict)
      this.nodes.delete(evict)
      this.vectors.delete(evict)
    }
  }

  async applyChanges(changes: PersistenceChanges): Promise<void> {
    if (changes.operationId && this.mutations.some(record => record.operationId === changes.operationId)) return
    if (changes.indexDefinitions) this.indexDefinitions = changes.indexDefinitions.map(index => ({ ...index, fields: [...index.fields] }))
    for (const id of changes.deleteNodeIds) { this.unindexNode(id); this.nodes.delete(id); this.nodeOrder.delete(id) }
    for (const id of changes.deleteEdgeIds) { this.unindexEdge(id); this.edges.delete(id) }
    for (const id of changes.deleteVectorIds) this.vectors.delete(id)
    for (const node of changes.putNodes) { this.indexNode(node); this.nodes.set(node.id, node); this.touchNode(node.id) }
    for (const edge of changes.putEdges) { this.edges.set(edge.id, edge); this.indexEdge(edge) }
    for (const entry of changes.putVectors) this.vectors.set(entry.id, entry.vector)
    this.evictIfOverCap()
    const record = mutationRecordFromChanges(changes, this.nextMutation++)
    if (record) this.mutations.push(record)
  }

  async getIndexDefinitions(): Promise<IndexDefinition[]> {
    return this.indexDefinitions.map(index => ({ ...index, fields: [...index.fields] }))
  }

  async getMutationsSince(sequence: bigint): Promise<MutationRecord[]> {
    return this.mutations.filter(record => record.sequence > sequence).map(record => ({ ...record, operations: record.operations.map(operation => ({ ...operation, payload: structuredClone(operation.payload) })) }))
  }

  async latestMutationSequence(): Promise<bigint> {
    return this.nextMutation - 1n
  }

  async stats() {
    return {
      persistedNodeCount: this.nodes.size,
      edgeCount: this.edges.size,
      vectorCount: this.vectors.size,
      mutationCount: this.mutations.length,
    }
  }

  async putNode(node: SerializedNode): Promise<void> {
    this.indexNode(node)
    this.nodes.set(node.id, node)
    this.touchNode(node.id)
    this.evictIfOverCap()
  }

  async bulkPutNodes(nodes: SerializedNode[]): Promise<void> {
    for (const node of nodes) { this.indexNode(node); this.nodes.set(node.id, node); this.touchNode(node.id) }
    this.evictIfOverCap()
  }

  async getNode(id: string): Promise<SerializedNode | undefined> {
    return this.nodes.get(id)
  }

  async getNodes(ids: string[]): Promise<SerializedNode[]> {
    return ids.map(id => this.nodes.get(id)).filter(Boolean) as SerializedNode[]
  }

  async deleteNode(id: string): Promise<void> {
    this.unindexNode(id)
    this.nodes.delete(id)
    this.nodeOrder.delete(id)
  }

  async bulkDeleteNodes(ids: string[]): Promise<void> {
    for (const id of ids) { this.unindexNode(id); this.nodes.delete(id); this.nodeOrder.delete(id) }
  }

  async allNodeIds(): Promise<string[]> {
    return [...this.nodes.keys()]
  }

  async queryNodes(query: PersistedNodeQuery): Promise<SerializedNode[]> {
    const candidates = this.typeCandidateIds(query)
    const nodes = candidates
      ? candidates.map(id => this.nodes.get(id)).filter((n): n is SerializedNode => !!n)
      : [...this.nodes.values()]
    return applyPersistedNodeQuery(nodes, query)
  }

  async countNodes(query: PersistedNodeQuery): Promise<number> {
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
  }

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

  async putEdge(edge: SerializedEdge): Promise<void> {
    this.edges.set(edge.id, edge)
    this.indexEdge(edge)
  }

  async bulkPutEdges(edges: SerializedEdge[]): Promise<void> {
    for (const edge of edges) { this.edges.set(edge.id, edge); this.indexEdge(edge) }
  }

  async getAllEdges(): Promise<SerializedEdge[]> {
    return [...this.edges.values()]
  }

  async getEdgesBySources(sources: string[], type?: string): Promise<SerializedEdge[]> {
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
  }

  async getEdgesByTargets(targets: string[], type?: string): Promise<SerializedEdge[]> {
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
  }

  async deleteEdge(id: string): Promise<void> {
    this.unindexEdge(id)
    this.edges.delete(id)
  }

  async bulkDeleteEdges(ids: string[]): Promise<void> {
    for (const id of ids) { this.unindexEdge(id); this.edges.delete(id) }
  }

  async putVector(id: string, vector: number[]): Promise<void> {
    this.vectors.set(id, vector)
  }

  async bulkPutVectors(entries: Array<{ id: string; vector: number[] }>): Promise<void> {
    for (const entry of entries) this.vectors.set(entry.id, entry.vector)
  }

  async deleteVector(id: string): Promise<void> {
    this.vectors.delete(id)
  }

  async getVectors(ids: string[]): Promise<Array<{ id: string; vector: number[] }>> {
    const results: Array<{ id: string; vector: number[] }> = []
    for (const id of ids) {
      const vector = this.vectors.get(id)
      if (vector) results.push({ id, vector })
    }
    return results
  }

  async getAllVectors(): Promise<Array<{ id: string; vector: number[] }>> {
    return [...this.vectors.entries()].map(([id, vector]) => ({ id, vector }))
  }

  async clearAll(): Promise<void> {
    this.nodes.clear()
    this.edges.clear()
    this.vectors.clear()
    this.byType.clear()
    this.edgesBySource.clear()
    this.edgesByTarget.clear()
  }

  async close(): Promise<void> {
    // No-op for in-memory
  }
}

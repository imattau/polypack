import type { SerializedNode, SerializedEdge } from '../types.js'
import type { PersistenceAdapter, PersistenceChanges, PersistedNodeQuery } from './adapter.js'
import { applyPersistedCountPagination, applyPersistedNodeQuery, matchesPersistedNode } from './query.js'

/** Volatile persistence adapter for Node.js, tests, and temporary graphs. */
export class MemoryAdapter implements PersistenceAdapter {
  private nodes = new Map<string, SerializedNode>()
  private edges = new Map<string, SerializedEdge>()
  private vectors = new Map<string, number[]>()
  private nodeOrder = new Map<string, true>()
  private readonly maxNodes: number | undefined

  constructor(maxNodes?: number) {
    this.maxNodes = maxNodes
  }

  private touchNode(id: string): void {
    this.nodeOrder.delete(id)
    this.nodeOrder.set(id, true)
  }

  private evictIfOverCap(): void {
    if (this.maxNodes === undefined) return
    while (this.nodeOrder.size > this.maxNodes) {
      const evict = this.nodeOrder.keys().next().value
      if (evict === undefined) break
      this.nodeOrder.delete(evict)
      this.nodes.delete(evict)
      this.vectors.delete(evict)
    }
  }

  async applyChanges(changes: PersistenceChanges): Promise<void> {
    for (const id of changes.deleteNodeIds) { this.nodes.delete(id); this.nodeOrder.delete(id) }
    for (const id of changes.deleteEdgeIds) this.edges.delete(id)
    for (const id of changes.deleteVectorIds) this.vectors.delete(id)
    for (const node of changes.putNodes) { this.nodes.set(node.id, node); this.nodeOrder.set(node.id, true) }
    for (const edge of changes.putEdges) this.edges.set(edge.id, edge)
    for (const entry of changes.putVectors) this.vectors.set(entry.id, entry.vector)
    this.evictIfOverCap()
  }

  async putNode(node: SerializedNode): Promise<void> {
    this.nodes.set(node.id, node)
    this.touchNode(node.id)
    this.evictIfOverCap()
  }

  async bulkPutNodes(nodes: SerializedNode[]): Promise<void> {
    for (const node of nodes) { this.nodes.set(node.id, node); this.touchNode(node.id) }
    this.evictIfOverCap()
  }

  async getNode(id: string): Promise<SerializedNode | undefined> {
    return this.nodes.get(id)
  }

  async getNodes(ids: string[]): Promise<SerializedNode[]> {
    return ids.map(id => this.nodes.get(id)).filter(Boolean) as SerializedNode[]
  }

  async deleteNode(id: string): Promise<void> {
    this.nodes.delete(id)
    this.nodeOrder.delete(id)
  }

  async bulkDeleteNodes(ids: string[]): Promise<void> {
    for (const id of ids) { this.nodes.delete(id); this.nodeOrder.delete(id) }
  }

  async allNodeIds(): Promise<string[]> {
    return [...this.nodes.keys()]
  }

  async queryNodes(query: PersistedNodeQuery): Promise<SerializedNode[]> {
    return applyPersistedNodeQuery([...this.nodes.values()], query)
  }

  async countNodes(query: PersistedNodeQuery): Promise<number> {
    let count = 0
    for (const node of this.nodes.values()) {
      if (matchesPersistedNode(node, query)) count++
    }
    return applyPersistedCountPagination(count, query)
  }

  async putEdge(edge: SerializedEdge): Promise<void> {
    this.edges.set(edge.id, edge)
  }

  async bulkPutEdges(edges: SerializedEdge[]): Promise<void> {
    for (const edge of edges) this.edges.set(edge.id, edge)
  }

  async getAllEdges(): Promise<SerializedEdge[]> {
    return [...this.edges.values()]
  }

  async getEdgesBySources(sources: string[], type?: string): Promise<SerializedEdge[]> {
    const sourceSet = new Set(sources)
    return [...this.edges.values()].filter(edge =>
      sourceSet.has(edge.source) && (type === undefined || edge.type === type)
    )
  }

  async getEdgesByTargets(targets: string[], type?: string): Promise<SerializedEdge[]> {
    const targetSet = new Set(targets)
    return [...this.edges.values()].filter(edge =>
      targetSet.has(edge.target) && (type === undefined || edge.type === type)
    )
  }

  async deleteEdge(id: string): Promise<void> {
    this.edges.delete(id)
  }

  async bulkDeleteEdges(ids: string[]): Promise<void> {
    for (const id of ids) this.edges.delete(id)
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
  }

  async close(): Promise<void> {
    // No-op for in-memory
  }
}

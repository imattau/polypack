import type { SerializedNode, SerializedEdge } from '../types.js'
import type { PersistenceAdapter, PersistedNodeQuery } from './adapter.js'
import { applyPersistedNodeQuery, matchesPersistedNode } from './query.js'

/** Volatile persistence adapter for Node.js, tests, and temporary graphs. */
export class MemoryAdapter implements PersistenceAdapter {
  private nodes = new Map<string, SerializedNode>()
  private edges = new Map<string, SerializedEdge>()
  private vectors = new Map<string, number[]>()

  async putNode(node: SerializedNode): Promise<void> {
    this.nodes.set(node.id, node)
  }

  async bulkPutNodes(nodes: SerializedNode[]): Promise<void> {
    for (const node of nodes) this.nodes.set(node.id, node)
  }

  async getNode(id: string): Promise<SerializedNode | undefined> {
    return this.nodes.get(id)
  }

  async getNodes(ids: string[]): Promise<SerializedNode[]> {
    return ids.map(id => this.nodes.get(id)).filter(Boolean) as SerializedNode[]
  }

  async deleteNode(id: string): Promise<void> {
    this.nodes.delete(id)
  }

  async bulkDeleteNodes(ids: string[]): Promise<void> {
    for (const id of ids) this.nodes.delete(id)
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
    return count
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

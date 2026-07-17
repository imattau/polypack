import type { SerializedNode, SerializedEdge } from '../types.js'

/** Storage-level node predicates used by persisted queries. */
export interface PersistedNodeQuery {
  nodeTypes?: string[]
  attributes?: Record<string, unknown>
  attributeRanges?: Record<string, { above?: number; below?: number }>
  orderBy?: { field: string; direction: 'asc' | 'desc' }
}

/** Storage contract used by {@link PolyGraph}. Implement all writes atomically where possible. */
export interface PersistenceAdapter {
  putNode(node: SerializedNode): Promise<void>
  bulkPutNodes(nodes: SerializedNode[]): Promise<void>
  getNode(id: string): Promise<SerializedNode | undefined>
  getNodes(ids: string[]): Promise<SerializedNode[]>
  deleteNode(id: string): Promise<void>
  bulkDeleteNodes(ids: string[]): Promise<void>
  allNodeIds(): Promise<string[]>
  /** Optional optimized query hook. Callers fall back to getNodes/allNodeIds. */
  queryNodes?(query: PersistedNodeQuery): Promise<SerializedNode[]>
  /** Optional optimized count hook for the same storage-level predicates. */
  countNodes?(query: PersistedNodeQuery): Promise<number>

  putEdge(edge: SerializedEdge): Promise<void>
  bulkPutEdges(edges: SerializedEdge[]): Promise<void>
  getAllEdges(): Promise<SerializedEdge[]>
  /** Optional indexed lookup used by persisted graph queries. */
  getEdgesBySources?(sources: string[], type?: string): Promise<SerializedEdge[]>
  /** Optional indexed reverse lookup used by persisted graph queries. */
  getEdgesByTargets?(targets: string[], type?: string): Promise<SerializedEdge[]>
  deleteEdge(id: string): Promise<void>
  bulkDeleteEdges(ids: string[]): Promise<void>

  putVector(id: string, vector: number[]): Promise<void>
  bulkPutVectors(entries: Array<{ id: string; vector: number[] }>): Promise<void>
  deleteVector(id: string): Promise<void>
  getAllVectors(): Promise<Array<{ id: string; vector: number[] }>>

  clearAll(): Promise<void>
  close(): Promise<void>
}

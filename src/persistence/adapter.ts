import type { SerializedNode, SerializedEdge } from '../types.js'

/** Storage contract used by {@link PolyGraph}. Implement all writes atomically where possible. */
export interface PersistenceAdapter {
  putNode(node: SerializedNode): Promise<void>
  bulkPutNodes(nodes: SerializedNode[]): Promise<void>
  getNode(id: string): Promise<SerializedNode | undefined>
  getNodes(ids: string[]): Promise<SerializedNode[]>
  deleteNode(id: string): Promise<void>
  bulkDeleteNodes(ids: string[]): Promise<void>
  allNodeIds(): Promise<string[]>

  putEdge(edge: SerializedEdge): Promise<void>
  bulkPutEdges(edges: SerializedEdge[]): Promise<void>
  getAllEdges(): Promise<SerializedEdge[]>
  deleteEdge(id: string): Promise<void>
  bulkDeleteEdges(ids: string[]): Promise<void>

  putVector(id: string, vector: number[]): Promise<void>
  bulkPutVectors(entries: Array<{ id: string; vector: number[] }>): Promise<void>
  deleteVector(id: string): Promise<void>
  getAllVectors(): Promise<Array<{ id: string; vector: number[] }>>

  clearAll(): Promise<void>
  close(): Promise<void>
}

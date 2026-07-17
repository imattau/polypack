/** A typed property-graph node. Timestamps are Unix milliseconds. */
export interface PolyNode<TData extends Record<string, unknown> = Record<string, unknown>> {
  id: string
  type: string
  data: TData
  vector?: Float64Array
  insertedAt: number
  updatedAt: number
}

/** Controls what happens to a target when its incoming edge is removed. */
export type EdgeOwnership = 'owned' | 'shared' | 'reference'

/** Serializable representation of a directed property-graph edge. */
export interface PolyEdge<TData extends Record<string, unknown> = Record<string, unknown>> {
  id: string
  source: string
  target: string
  type: string
  data?: TData
  createdAt: number
}

/** Mutation notification emitted through {@link PolyGraph.changes}. */
export interface GraphChangeEvent {
  type: 'node_added' | 'node_updated' | 'node_removed' | 'edge_added' | 'edge_removed'
  nodeId?: string
  nodeType?: string
  edgeId?: string
  edgeType?: string
  source?: string
  target?: string
}

/** Persistence-safe node representation using plain arrays for vectors. */
export interface SerializedNode {
  id: string
  type: string
  data: Record<string, unknown>
  vector: number[] | null
  insertedAt: number
  updatedAt: number
}

/** Persistence-safe edge representation. */
export interface SerializedEdge {
  id: string
  source: string
  target: string
  type: string
  data: Record<string, unknown> | null
  createdAt: number
}

/** Parameters for a nearest-neighbour vector search. */
export interface VectorQuery {
  vector: number[]
  threshold?: number
  topK: number
}

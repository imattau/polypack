export interface PolyNode<TData extends Record<string, unknown> = Record<string, unknown>> {
  id: string
  type: string
  data: TData
  vector?: Float64Array
  insertedAt: number
  updatedAt: number
}

export type EdgeOwnership = 'owned' | 'shared' | 'reference'

export interface PolyEdge<TData extends Record<string, unknown> = Record<string, unknown>> {
  id: string
  source: string
  target: string
  type: string
  data?: TData
  createdAt: number
}

export interface GraphChangeEvent {
  type: 'node_added' | 'node_updated' | 'node_removed' | 'edge_added' | 'edge_removed'
  nodeId?: string
  nodeType?: string
  edgeId?: string
  edgeType?: string
  source?: string
  target?: string
}

export interface SerializedNode {
  id: string
  type: string
  data: Record<string, unknown>
  vector: number[] | null
  insertedAt: number
  updatedAt: number
}

export interface SerializedEdge {
  id: string
  source: string
  target: string
  type: string
  data: Record<string, unknown> | null
  createdAt: number
}

export interface VectorQuery {
  vector: number[]
  threshold?: number
  topK: number
}

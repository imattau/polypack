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

/**
 * Optional transform for handling non-cloneable data (Blob, File, etc.) that
 * cannot pass through `structuredClone`. Return `sidecar` from `serialize` —
 * it is stored in memory alongside the node and re-supplied to `deserialize`
 * on every read. The sidecar is NOT persisted across sessions.
 */
export interface DataTransform {
  serialize?(data: Record<string, unknown>): { data: Record<string, unknown>; sidecar?: unknown }
  deserialize?(data: Record<string, unknown>, sidecar?: unknown): Record<string, unknown>
}

/** Infer the edge-type value type from a definition object. */
export type EdgeTypes<T extends Record<string, string>> = { readonly [K in keyof T]: T[K] }

/**
 * Create a frozen edge-type constant object with full type inference.
 *
 * ```ts
 * export const EDGE = defineEdges({
 *   HAS_LABEL: 'HAS_LABEL',
 *   REPLIES_TO: 'REPLIES_TO',
 * })
 * // typeof EDGE.HAS_LABEL === 'HAS_LABEL'  (literal type)
 * ```
 */
export function defineEdges<T extends Record<string, string>>(def: T): EdgeTypes<T> {
  return Object.freeze({ ...def }) as EdgeTypes<T>
}

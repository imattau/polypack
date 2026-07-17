export { PolyGraph } from './graph'
export { VectorIndex, cosineSimilarity, euclideanSimilarity } from './vector-index'
export type { DistanceFunction } from './vector-index'
export { GraphQuery } from './query'
export type { AggregateOp, AggregateResult, GroupedRow } from './query'
export { MemoryAdapter } from './persistence/memory'
export { IndexedDBAdapter } from './persistence/indexeddb'
export type { PersistenceAdapter } from './persistence/adapter'
export type { IndexedDBConfig } from './persistence/indexeddb'
export type {
  PolyNode,
  PolyEdge,
  EdgeOwnership,
  GraphChangeEvent,
  SerializedNode,
  SerializedEdge,
  VectorQuery,
} from './types'
export { yieldToUI, edgeId } from './utils'

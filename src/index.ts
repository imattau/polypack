export { PolyGraph } from './graph.js'
export { VectorIndex, cosineSimilarity, euclideanSimilarity } from './vector-index.js'
export type { DistanceFunction } from './vector-index.js'
export { FeatureHashEmbedding, createEmbedding, defaultEmbedding } from './embedding.js'
export type { EmbeddingProvider, EmbeddingVector, FeatureHashEmbeddingOptions } from './embedding.js'
export { GraphQuery } from './query.js'
export type { AggregateOp, AggregateResult, GroupedRow } from './query.js'
export { PersistedGraphQuery } from './persisted-query.js'
export { MemoryAdapter } from './persistence/memory.js'
export { IndexedDBAdapter } from './persistence/indexeddb.js'
export type { PersistenceAdapter, PersistenceChanges, PersistedNodeQuery } from './persistence/adapter.js'
export type { IndexedDBConfig } from './persistence/indexeddb.js'
export type {
  PolyNode,
  PolyEdge,
  EdgeOwnership,
  GraphChangeEvent,
  SerializedNode,
  SerializedEdge,
  VectorQuery,
} from './types.js'
export { yieldToUI, edgeId } from './utils.js'

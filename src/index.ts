export { PolyGraph } from './graph.js'
export { VectorIndex, cosineSimilarity, euclideanSimilarity } from './vector-index.js'
export { HNSWIndex } from './hnsw-index.js'
export type { HNSWConfig } from './hnsw-index.js'
export type { DistanceFunction, VectorIndexLike } from './vector-index.js'
export { FeatureHashEmbedding, createEmbedding, defaultEmbedding, buildEmbeddingText } from './embedding.js'
export type { EmbeddingProvider, EmbeddingVector, FeatureHashEmbeddingOptions } from './embedding.js'
export { GraphQuery } from './query.js'
export type { AggregateOp, AggregateResult, GroupedRow } from './query.js'
export { PersistedGraphQuery } from './persisted-query.js'
export { MemoryAdapter } from './persistence/memory.js'
export type { PersistenceAdapter, PersistenceChanges, PersistedNodeQuery } from './persistence/adapter.js'
export type {
  PolyNode,
  PolyEdge,
  EdgeOwnership,
  GraphChangeEvent,
  SerializedNode,
  SerializedEdge,
  VectorQuery,
  EdgeTypes,
  DataTransform,
} from './types.js'
export { defineEdges } from './types.js'
export { yieldToUI, edgeId } from './utils.js'

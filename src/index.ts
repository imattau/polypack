export { PolyGraph, GraphSnapshot } from './graph.js'
export { ConflictError } from './errors.js'
export { UniqueConstraintError } from './index-errors.js'
export { SchemaValidationError } from './schema-errors.js'
export { MigrationError, MigrationRegistry } from './migrations.js'
export { ReadOnlyStoreError, StoreLockError } from './persistence/lock-errors.js'
export { ResourceLimitError } from './resource-errors.js'
export { AdapterCapabilityError } from './capability-errors.js'
export { QueryLimitError, QueryAbortedError } from './query-errors.js'
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
export { ActivationEngine, mergeActivation } from './activation.js'
export type { ActivationConfig, SpreadOptions, PulseOptions, VectorLike } from './activation.js'
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
  NodeActivation,
  WriteOptions,
  NodePatch,
  AdapterCapabilities,
  GraphTransaction,
  IndexDefinition,
  QueryExplain,
  GraphOperation,
  MutationRecord,
  VerificationReport,
  GraphStats,
  QueryResourceLimits,
  QueryMetrics,
  GraphResourceLimits,
  EdgeCardinality,
  NodeTypeDefinition,
  EdgeTypeDefinition,
} from './types.js'
export type { MigrationDefinition, MigrationOptions, MigrationProgress, MigrationReport } from './migrations.js'
export { defineEdges } from './types.js'
export { yieldToUI, edgeId } from './utils.js'

/**
 * Type declarations for the polypack-native wrapper.
 *
 * `NativeVectorIndex` and `NativeHnswIndex` are drop-in replacements for the
 * TypeScript `VectorIndex` / `HNSWIndex` classes backed by the Rust core.
 */

export type DistanceFunction = (a: ArrayLike<number>, b: ArrayLike<number>) => number

export interface ScoredId {
  id: string
  score: number
}

export interface EngineInfo {
  graph: string
  vector: string
  storage: string
  available: boolean
}

/** True when the native addon for this platform loaded successfully. */
export function isNativeAvailable(): boolean

/** Report which engine is active (rust-native, or the TypeScript fallback). */
export function engineInfo(): EngineInfo

/** Detect the engine currently used for a graph, for diagnostics. */
export function detectEngine(): string

/** Drop-in replacement for the TypeScript `VectorIndex`. */
export class NativeVectorIndex {
  constructor(onChange?: (id: string) => void, distance?: 'cosine' | 'euclidean')
  add(id: string, vector: number[] | Float64Array): void
  hydrate(id: string, vector: number[] | Float64Array): void
  addMany(entries: Array<{ id: string; vector: number[] | Float64Array }>): void
  remove(id: string): void
  removeMany(ids: string[]): void
  query(vector: number[], topK: number, threshold?: number): ScoredId[]
  clear(): void
  readonly size: number
  entries(): IterableIterator<[string, Float64Array]>
  has(id: string): boolean
  get(id: string): Float64Array | undefined
}

export interface NativeHnswConfig {
  M?: number
  Mmax0?: number
  efConstruction?: number
  efSearch?: number
}

/** Drop-in replacement for the TypeScript `HNSWIndex` (cosine distance only). */
export class NativeHnswIndex {
  constructor(onChange?: (id: string) => void, distanceFn?: DistanceFunction, config?: NativeHnswConfig)
  add(id: string, vector: number[] | Float64Array): void
  update(id: string, vector: number[] | Float64Array): void
  hydrate(id: string, vector: number[] | Float64Array): void
  addMany(entries: Array<{ id: string; vector: number[] | Float64Array }>): void
  remove(id: string): void
  removeMany(ids: string[]): void
  query(vector: number[], topK: number, threshold?: number): ScoredId[]
  clear(): void
  readonly size: number
  entries(): IterableIterator<[string, Float64Array]>
  has(id: string): boolean
  get(id: string): Float64Array | undefined
}

/** Factory for `PolyGraph`'s `createVectorIndex` hook using the native engine. */
export function createNativeVectorIndex(): (onChange: (id: string) => void) => NativeVectorIndex

/** Factory returning a native HNSW index. */
export function createNativeHnswIndex(onChange?: (id: string) => void): NativeHnswIndex

export interface NativeChangeBatch {
  putNodes?: Array<Record<string, unknown>>
  deleteNodeIds?: string[]
  putEdges?: Array<Record<string, unknown>>
  deleteEdgeIds?: string[]
  putVectors?: Array<{ id: string; vector: number[] }>
  deleteVectorIds?: string[]
}

/**
 * Directory-backed durable store over the Rust persistence state machine.
 * Files (`snapshot.msgpack`, `wal.msgpack`) are byte-compatible with the
 * TypeScript `BinaryStoreAdapter`.
 */
export class NativeStore {
  constructor(directory: string, compactThreshold?: number)
  apply(changes: NativeChangeBatch): void
  nodeIds(): string[]
  getNode(id: string): Record<string, unknown> | undefined
  allEdges(): Array<Record<string, unknown>>
  allVectors(): Array<[string, number[]]>
  compact(): void
  close(): void
}

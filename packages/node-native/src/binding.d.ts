/**
 * Runtime types for the loaded `.node` addon (generated bindings surface from
 * `crates/polypack-node`). The loader in index.ts selects the platform binary.
 */

export interface NativeScoredId {
  id: string
  score: number
}

export interface NativeIndexEntry {
  id: string
  vector: Float64Array
}

export interface NativeHnswConfig {
  m?: number
  mmax0?: number
  efConstruction?: number
  efSearch?: number
}

export interface EngineInfo {
  graph: string
  vector: string
  storage: string
}

export interface NativeExactIndexBinding {
  constructor(distance?: string): NativeExactIndexBinding
  add(id: string, vector: Float64Array): void
  addMany(ids: string[], vectors: Float64Array[]): void
  remove(id: string): void
  removeMany(ids: string[]): void
  query(vector: Float64Array, topK: number, threshold?: number): NativeScoredId[]
  clear(): void
  has(id: string): boolean
  get(id: string): Float64Array | undefined
  entries(): NativeIndexEntry[]
  readonly size: number
}

export interface NativeHnswIndexBinding {
  constructor(config?: NativeHnswConfig, levelSeed?: number): NativeHnswIndexBinding
  add(id: string, vector: Float64Array): void
  update(id: string, vector: Float64Array): void
  addMany(ids: string[], vectors: Float64Array[]): void
  remove(id: string): void
  removeMany(ids: string[]): void
  query(vector: Float64Array, topK: number, threshold?: number): NativeScoredId[]
  clear(): void
  has(id: string): boolean
  get(id: string): Float64Array | undefined
  entries(): NativeIndexEntry[]
  readonly size: number
}

export interface NativeBinding {
  NativeExactIndex: new (distance?: string) => NativeExactIndexBinding
  NativeHnswIndex: new (config?: NativeHnswConfig, levelSeed?: number) => NativeHnswIndexBinding
  NativeStore: new (directory: string, compactThreshold?: number) => NativeStoreBinding
  engineInfo(): EngineInfo
}

export interface NativeChangeBatch {
  putNodes?: unknown[]
  deleteNodeIds?: string[]
  putEdges?: unknown[]
  deleteEdgeIds?: string[]
  putVectors?: Array<{ id: string; vector: number[] }>
  deleteVectorIds?: string[]
}

export interface NativeStoreBinding {
  apply(changes: NativeChangeBatch): void
  nodeIds(): string[]
  getNode(id: string): Record<string, unknown> | undefined
  allEdges(): Array<Record<string, unknown>>
  allVectors(): Array<[string, number[]]>
  compact(): void
  close(): void
}

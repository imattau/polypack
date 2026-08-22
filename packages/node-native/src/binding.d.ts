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
  /** `"cosine"` (default) or `"euclidean"`. */
  distance?: string
}

export interface NativeNodeActivation {
  score: number
  importance: number
  reinforcementCount: number
  lastMeaningfulActivation: number
  /** Suppression, subtracted from `score` at read time only. Absent is equivalent to 0. */
  inhibition?: number
  /** Epoch-ms anchor for `inhibition`'s decay. Absent iff `inhibition` is absent. */
  lastInhibitedAt?: number
  /** Per-context activation, additional to (not a replacement for) the global `score` above. */
  context?: Record<string, { score: number; lastMeaningfulActivation: number }>
}

export interface EngineInfo {
  graph: string
  vector: string
  storage: string
}

export interface NativeActivationScoreBreakdown {
  semantic: number
  graph: number
  recency: number
  usage: number
  weightedSemantic: number
  weightedGraph: number
  weightedRecency: number
  weightedUsage: number
  total: number
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
  NativeStore: new (directory: string, compactThreshold?: number, readOnly?: boolean) => NativeStoreBinding
  restoreStore(source: string, destination: string, compactThreshold?: number): NativeStoreBinding
  engineInfo(): EngineInfo
  executeQueryPlan(nodes: unknown[], edges: unknown[], plan: Record<string, unknown>): string[]
  aggregateQueryPlan(
    nodes: unknown[],
    edges: unknown[],
    plan: Record<string, unknown>,
    field: string,
    op: string,
  ): { value: number; count: number }
  decayFactor(elapsedMs: number, halfLifeMs: number): number
  mergeActivation(
    existing: NativeNodeActivation,
    incoming: NativeNodeActivation,
    now?: number,
  ): NativeNodeActivation
  reinforceActivation(
    previous: NativeNodeActivation | undefined,
    delta: number,
    now: number,
    context?: string,
  ): NativeNodeActivation
  suppressActivation(
    previous: NativeNodeActivation | undefined,
    delta: number,
    now: number,
  ): NativeNodeActivation
  activationScoreOf(score: number, lastMeaningfulActivation: number, now: number, halfLifeMs: number): number
  estimateNodeTokens(serializedMemory: string): number
  scoreBreakdown(
    semantic: number,
    graph: number,
    recency: number,
    usage: number,
    semanticWeight?: number,
    graphWeight?: number,
    recencyWeight?: number,
    usageWeight?: number,
  ): NativeActivationScoreBreakdown
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
  nodeCount(): number
  queryNodes(query: Record<string, unknown>): Array<Record<string, unknown>>
  countNodes(query: Record<string, unknown>): number
  defineIndex(definition: Record<string, unknown>): void
  dropIndex(name: string): boolean
  indexDefinitions(): Array<Record<string, unknown>>
  registerNodeType(definition: Record<string, unknown>): void
  registerEdgeType(definition: Record<string, unknown>): void
  getEdgesBySources(sources: string[], edgeType?: string): Array<Record<string, unknown>>
  getEdgesByTargets(targets: string[], edgeType?: string): Array<Record<string, unknown>>
  getNode(id: string): Record<string, unknown> | undefined
  allEdges(): Array<Record<string, unknown>>
  allVectors(): Array<[string, number[]]>
  compact(): void
  checkpoint(): void
  verify(): Record<string, unknown>
  capabilities(): Record<string, unknown>
  stats(): Record<string, unknown>
  mutationLogSince(sequence: string, limit?: number): Array<Record<string, unknown>>
  latestMutationSequence(): string
  backup(destination: string): void
  close(): void
}

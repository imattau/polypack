/**
 * Native polypack-core bindings for Node.js.
 *
 * Loads the platform `.node` addon built from `crates/polypack-node` and
 * exposes drop-in replacements for the TypeScript `VectorIndex` and
 * `HNSWIndex` classes, plus factories that satisfy `PolyGraph`'s
 * `createVectorIndex` hook. When the binary is absent the loader falls back
 * cleanly and `engineInfo()` reports the TypeScript engine.
 */
import { createRequire } from 'node:module'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { cosineSimilarity as cosine, euclideanSimilarity as euclidean } from '../../../src/vector-index.js'
import type { DistanceFunction } from '../../../src/vector-index.js'
import { setNativeQueryExecutor, isNativeQueryExecutorActive } from '../../../src/query.js'
import type {
  EngineInfo as BindingEngineInfo,
  NativeBinding,
  NativeExactIndexBinding,
  NativeHnswIndexBinding,
  NativeNodeActivation,
  NativeActivationScoreBreakdown,
  NativeStoreBinding,
} from './binding.js'

const require = createRequire(import.meta.url)
const THIS_DIR = dirname(fileURLToPath(import.meta.url))

const TRIPLES: Record<string, string> = {
  'linux-x64': 'linux-x64-gnu',
  'linux-arm64': 'linux-arm64-gnu',
  'darwin-x64': 'darwin-x64',
  'darwin-arm64': 'darwin-arm64',
  'win32-x64': 'win32-x64-msvc',
}

/** Published per-platform package names (optionalDependencies). */
const PLATFORM_PACKAGES: Record<string, string> = {
  'linux-x64-gnu': '@0xx0lostcause0xx0/polypack-native-linux-x64-gnu',
  'linux-arm64-gnu': '@0xx0lostcause0xx0/polypack-native-linux-arm64-gnu',
  'darwin-x64': '@0xx0lostcause0xx0/polypack-native-darwin-x64',
  'darwin-arm64': '@0xx0lostcause0xx0/polypack-native-darwin-arm64',
  'win32-x64-msvc': '@0xx0lostcause0xx0/polypack-native-win32-x64-msvc',
}

function loadNative(): NativeBinding | null {
  try {
    const triple = TRIPLES[`${process.platform}-${process.arch}`]
    if (!triple) return null
    const name = `polypack-native.${triple}.node`
    // Prefer a local monorepo build during development. Published packages do
    // not ship this file, so they naturally fall through to the platform pkg.
    const candidates = [
      join(THIS_DIR, '..', 'dist', name),
      join(THIS_DIR, name),
    ]
    for (const candidate of candidates) {
      try {
        return require(candidate) as NativeBinding
      } catch {
        // try the next candidate
      }
    }
    // Installed platform package (published distribution).
    const platformPkg = PLATFORM_PACKAGES[triple]
    if (platformPkg) {
      try {
        return require(platformPkg) as NativeBinding
      } catch {
        // not installed
      }
    }
    return null
  } catch {
    return null
  }
}

const native: NativeBinding | null = loadNative()

export interface EngineInfo extends BindingEngineInfo {
  available: boolean
  query: 'rust-native' | 'typescript'
}

/** True when the native addon for this platform is loadable. */
export function isNativeAvailable(): boolean {
  return native !== null
}

/**
 * Report which engine is active: `rust-native` when the addon loaded,
 * otherwise the TypeScript fallback. `query` reflects the GraphQuery path.
 */
export function engineInfo(): EngineInfo {
  if (!native) {
    return {
      graph: 'typescript',
      vector: 'typescript',
      storage: 'host',
      available: false,
      query: 'typescript',
    }
  }
  return {
    ...native.engineInfo(),
    available: true,
    query: isNativeQueryExecutorActive() ? 'rust-native' : 'typescript',
  }
}

/** Detect the engine currently used for a graph, for diagnostics. */
export function detectEngine(): string {
  return native ? 'rust-native' : 'typescript'
}

function assertAvailable(): asserts native is NativeBinding {
  if (!native) {
    throw new Error(
      'polypack-native binary is not available for this platform; ' +
        'install the correct package or use the TypeScript vector engine',
    )
  }
}

/** Re-throw native errors as the TS error classes from errors.md. */
function translateError(err: unknown): Error {
  const msg = err instanceof Error ? err.message : String(err)
  if (msg.startsWith('invalid_argument')) return new TypeError(msg)
  if (msg.startsWith('dimension_mismatch')) return new RangeError(msg)
  if (msg.startsWith('range_out_of_bounds')) return new RangeError(msg)
  return err instanceof Error ? err : new Error(msg)
}

function callNative<T>(fn: () => T): T {
  try {
    return fn()
  } catch (err) {
    throw translateError(err)
  }
}

function toFloat64(vector: number[] | Float64Array): Float64Array {
  return vector instanceof Float64Array ? vector : new Float64Array(vector)
}

export interface ScoredId {
  id: string
  score: number
}

/**
 * Drop-in replacement for the TypeScript `VectorIndex`, backed by the Rust
 * exact index. Validates inputs identically and invokes `onChange` for
 * mutations like the TypeScript implementation.
 */
export class NativeVectorIndex {
  private inner: NativeExactIndexBinding
  private onChange?: (id: string) => void

  constructor(onChange?: (id: string) => void, distance: 'cosine' | 'euclidean' = 'cosine') {
    assertAvailable()
    this.onChange = onChange
    this.inner = new native.NativeExactIndex(distance)
  }

  add(id: string, vector: number[] | Float64Array): void {
    if (!id) throw new TypeError('Vector id must not be empty')
    assertFiniteVector(vector)
    callNative(() => this.inner.add(id, toFloat64(vector)))
    this.onChange?.(id)
  }

  hydrate(id: string, vector: number[] | Float64Array): void {
    if (!id) throw new TypeError('Vector id must not be empty')
    assertFiniteVector(vector)
    callNative(() => this.inner.add(id, toFloat64(vector)))
  }

  addMany(entries: Array<{ id: string; vector: number[] | Float64Array }>): void {
    const ids: string[] = []
    const vectors: Float64Array[] = []
    for (const { id, vector } of entries) {
      if (!id) throw new TypeError('Vector id must not be empty')
      assertFiniteVector(vector)
      ids.push(id)
      vectors.push(toFloat64(vector))
    }
    callNative(() => this.inner.addMany(ids, vectors))
    for (const { id } of entries) this.onChange?.(id)
  }

  remove(id: string): void {
    this.inner.remove(id)
  }

  removeMany(ids: string[]): void {
    this.inner.removeMany(ids)
  }

  query(vector: number[], topK: number, threshold = 0): ScoredId[] {
    assertFiniteVector(vector, 'query vector')
    assertNonNegativeInteger(topK, 'topK')
    if (!Number.isFinite(threshold)) throw new RangeError('threshold must be finite')
    if (topK === 0) return []
    return callNative(() => this.inner.query(toFloat64(vector), topK, threshold))
  }

  clear(): void {
    this.inner.clear()
  }

  get size(): number {
    return this.inner.size
  }

  *entries(): IterableIterator<[string, Float64Array]> {
    for (const { id, vector } of this.inner.entries()) yield [id, vector]
  }

  has(id: string): boolean {
    return this.inner.has(id)
  }

  get(id: string): Float64Array | undefined {
    return this.inner.get(id)
  }
}

/**
 * Drop-in replacement for the TypeScript `HNSWIndex`, backed by the Rust
 * update-safe HNSW. Cosine and Euclidean distance are supported natively
 * (matching `cosineSimilarity`/`euclideanSimilarity` from `vector-index.js`);
 * passing any other custom distance function throws rather than silently
 * diverging.
 */
export class NativeHnswIndex {
  private inner: NativeHnswIndexBinding
  private onChange?: (id: string) => void

  constructor(
    onChange?: (id: string) => void,
    distanceFn?: DistanceFunction,
    config?: { M?: number; Mmax0?: number; efConstruction?: number; efSearch?: number },
  ) {
    assertAvailable()
    if (distanceFn && distanceFn !== cosine && distanceFn !== euclidean) {
      throw new Error('polypack-native HNSW only supports the cosine and euclidean distance functions')
    }
    this.onChange = onChange
    this.inner = new native.NativeHnswIndex(
      {
        m: config?.M,
        mmax0: config?.Mmax0,
        efConstruction: config?.efConstruction,
        efSearch: config?.efSearch,
        distance: distanceFn === euclidean ? 'euclidean' : 'cosine',
      },
      7,
    )
  }

  add(id: string, vector: number[] | Float64Array): void {
    if (!id) throw new TypeError('Vector id must not be empty')
    assertFiniteVector(vector)
    callNative(() => this.inner.add(id, toFloat64(vector)))
    this.onChange?.(id)
  }

  update(id: string, vector: number[] | Float64Array): void {
    if (!id) throw new TypeError('Vector id must not be empty')
    assertFiniteVector(vector)
    callNative(() => this.inner.update(id, toFloat64(vector)))
    this.onChange?.(id)
  }

  hydrate(id: string, vector: number[] | Float64Array): void {
    if (!id) throw new TypeError('Vector id must not be empty')
    assertFiniteVector(vector)
    callNative(() => this.inner.add(id, toFloat64(vector)))
  }

  addMany(entries: Array<{ id: string; vector: number[] | Float64Array }>): void {
    const ids: string[] = []
    const vectors: Float64Array[] = []
    for (const { id, vector } of entries) {
      if (!id) throw new TypeError('Vector id must not be empty')
      assertFiniteVector(vector)
      ids.push(id)
      vectors.push(toFloat64(vector))
    }
    callNative(() => this.inner.addMany(ids, vectors))
    for (const { id } of entries) this.onChange?.(id)
  }

  remove(id: string): void {
    this.inner.remove(id)
  }

  removeMany(ids: string[]): void {
    this.inner.removeMany(ids)
  }

  query(vector: number[], topK: number, threshold = 0): ScoredId[] {
    assertFiniteVector(vector, 'query vector')
    assertNonNegativeInteger(topK, 'topK')
    if (!Number.isFinite(threshold)) throw new RangeError('threshold must be finite')
    if (topK === 0) return []
    return callNative(() => this.inner.query(toFloat64(vector), topK, threshold))
  }

  clear(): void {
    this.inner.clear()
  }

  get size(): number {
    return this.inner.size
  }

  *entries(): IterableIterator<[string, Float64Array]> {
    for (const { id, vector } of this.inner.entries()) yield [id, vector]
  }

  has(id: string): boolean {
    return this.inner.has(id)
  }

  get(id: string): Float64Array | undefined {
    return this.inner.get(id)
  }
}

/** Factory for `PolyGraph`'s `createVectorIndex` hook using the native engine. */
export function createNativeVectorIndex(): (onChange: (id: string) => void) => NativeVectorIndex {
  return (onChange) => new NativeVectorIndex(onChange)
}

/** Factory returning a native HNSW index. */
export function createNativeHnswIndex(onChange?: (id: string) => void): NativeHnswIndex {
  return new NativeHnswIndex(onChange)
}

/**
 * Execute a query plan over serialized nodes/edges with the Rust query
 * executor, returning ordered node ids. `plan` follows the shared
 * query-plan IR (`specification/query-plan.schema.json`).
 */
export function executeQueryPlan(
  nodes: Array<Record<string, unknown>>,
  edges: Array<Record<string, unknown>>,
  plan: Record<string, unknown>,
): string[] {
  assertAvailable()
  return callNative(() => native.executeQueryPlan(nodes, edges, plan))
}

/**
 * Route `GraphQuery` (in-memory) through the Rust query executor when the
 * binary is present. Queries with join predicates fall back to TypeScript.
 */
export function installNativeQueryExecutor(): void {
  setNativeQueryExecutor((plan, nodes, edges) => {
    try {
      return native ? native.executeQueryPlan(nodes, edges, plan) : null
    } catch {
      return null
    }
  })
}

/** Aggregate a numeric field over the nodes matched by a query plan. */
export function aggregateQueryPlan(
  nodes: Array<Record<string, unknown>>,
  edges: Array<Record<string, unknown>>,
  plan: Record<string, unknown>,
  field: string,
  op: string,
): { value: number; count: number } {
  assertAvailable()
  return callNative(() => native.aggregateQueryPlan(nodes, edges, plan, field, op))
}

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
  private inner: NativeStoreBinding

  constructor(directory: string, compactThreshold?: number, readOnly = false) {
    assertAvailable()
    this.inner = new native.NativeStore(directory, compactThreshold, readOnly)
  }

  /** Restore and validate a native store from a directory backup. */
  static restore(source: string, destination: string, compactThreshold?: number): NativeStore {
    assertAvailable()
    const restored = Object.create(NativeStore.prototype) as NativeStore
    restored.inner = callNative(() => native.restoreStore(source, destination, compactThreshold))
    return restored
  }

  apply(changes: NativeChangeBatch): void {
    callNative(() => this.inner.apply(changes))
  }

  nodeIds(): string[] {
    return callNative(() => this.inner.nodeIds())
  }

  /** Total persisted node count, without materialising ids. */
  nodeCount(): number {
    return callNative(() => this.inner.nodeCount())
  }

  /** Query persisted nodes with a `PersistedNodeQuery`-shaped object. */
  queryNodes(query: Record<string, unknown>): Array<Record<string, unknown>> {
    return callNative(() => this.inner.queryNodes(query))
  }

  /** Define or replace a persisted node-data index. */
  defineIndex(definition: Record<string, unknown>): void {
    callNative(() => this.inner.defineIndex(definition))
  }

  /** Drop a persisted node-data index. */
  dropIndex(name: string): boolean {
    return callNative(() => this.inner.dropIndex(name))
  }

  /** Return persisted index definitions. */
  indexDefinitions(): Array<Record<string, unknown>> {
    return callNative(() => this.inner.indexDefinitions())
  }

  /** Register or replace a node-type schema definition. */
  registerNodeType(definition: Record<string, unknown>): void {
    callNative(() => this.inner.registerNodeType(definition))
  }

  /** Register or replace an edge-type schema definition. */
  registerEdgeType(definition: Record<string, unknown>): void {
    callNative(() => this.inner.registerEdgeType(definition))
  }

  /** Count persisted nodes matching a `PersistedNodeQuery`-shaped object. */
  countNodes(query: Record<string, unknown>): number {
    return callNative(() => this.inner.countNodes(query))
  }

  /** Edges from the given sources, optionally filtered by edge type. */
  getEdgesBySources(sources: string[], edgeType?: string): Array<Record<string, unknown>> {
    return callNative(() => this.inner.getEdgesBySources(sources, edgeType))
  }

  /** Edges targeting the given nodes, optionally filtered by edge type. */
  getEdgesByTargets(targets: string[], edgeType?: string): Array<Record<string, unknown>> {
    return callNative(() => this.inner.getEdgesByTargets(targets, edgeType))
  }

  getNode(id: string): Record<string, unknown> | undefined {
    return callNative(() => this.inner.getNode(id))
  }

  allEdges(): Array<Record<string, unknown>> {
    return callNative(() => this.inner.allEdges())
  }

  allVectors(): Array<[string, number[]]> {
    return callNative(() => this.inner.allVectors())
  }

  compact(): void {
    callNative(() => this.inner.compact())
  }

  /** Force a durable checkpoint and compact the recovery WAL. */
  checkpoint(): void {
    callNative(() => this.inner.checkpoint())
  }

  /** Validate persisted framing, records, references, vectors, and indexes. */
  verify(): {
    ok: boolean
    errors: string[]
    nodeCount: number
    edgeCount: number
    vectorCount: number
    mutationCount: number
  } {
    return callNative(() => this.inner.verify()) as {
      ok: boolean
      errors: string[]
      nodeCount: number
      edgeCount: number
      vectorCount: number
      mutationCount: number
    }
  }

  /** Report the guarantees provided by the native filesystem adapter. */
  capabilities(): {
    atomicBatches: boolean
    transactions: boolean
    fsync: boolean
    secondaryIndexes: boolean
    snapshots: boolean
    changeFeed: boolean
    concurrentWriters: boolean
    vectorSearch: 'none' | 'exact' | 'ann'
  } {
    return callNative(() => this.inner.capabilities()) as {
      atomicBatches: boolean
      transactions: boolean
      fsync: boolean
      secondaryIndexes: boolean
      snapshots: boolean
      changeFeed: boolean
      concurrentWriters: boolean
      vectorSearch: 'none' | 'exact' | 'ann'
    }
  }

  /** Return operational counters for the native persistence store. */
  stats(): {
    persistedNodeCount: number
    edgeCount: number
    vectorCount: number
    mutationCount: number
    walBytes: number
  } {
    return callNative(() => this.inner.stats()) as {
      persistedNodeCount: number
      edgeCount: number
      vectorCount: number
      mutationCount: number
      walBytes: number
    }
  }

  /** Read durable logical mutations after a sequence cursor. */
  mutationLogSince(sequence: bigint, limit?: number): Array<Record<string, unknown>> {
    return callNative(() => this.inner.mutationLogSince(sequence.toString(), limit))
  }

  /** Return the highest durable logical mutation sequence. */
  latestMutationSequence(): bigint {
    return BigInt(callNative(() => this.inner.latestMutationSequence()))
  }

  /** Create a consistent checkpointed directory backup. */
  backup(destination: string): void {
    callNative(() => this.inner.backup(destination))
  }

  close(): void {
    callNative(() => this.inner.close())
  }
}

function assertFiniteVector(vector: ArrayLike<number>, name = 'vector'): void {
  if (vector.length === 0) throw new RangeError(`${name} must not be empty`)
  for (let i = 0; i < vector.length; i++) {
    if (!Number.isFinite(vector[i])) throw new RangeError(`${name} must contain finite values`)
  }
}

export interface NodeActivationLike {
  score: number
  importance: number
  reinforcementCount: number
  lastMeaningfulActivation: number
  inhibition?: number
  lastInhibitedAt?: number
  context?: Record<string, { score: number; lastMeaningfulActivation: number }>
}

function toNativeActivation(a: NodeActivationLike): NativeNodeActivation {
  return {
    score: a.score,
    importance: a.importance,
    reinforcementCount: a.reinforcementCount,
    lastMeaningfulActivation: a.lastMeaningfulActivation,
    inhibition: a.inhibition,
    lastInhibitedAt: a.lastInhibitedAt,
    context: a.context,
  }
}

/** Exponential-decay multiplier `0.5 ** (elapsed / halfLife)`. */
export function decayFactor(elapsedMs: number, halfLifeMs: number): number {
  assertAvailable()
  return callNative(() => native.decayFactor(elapsedMs, halfLifeMs))
}

/** Merge two durable activation totals (max-merge, re-anchored to `now`). */
export function mergeActivation(
  existing: NodeActivationLike,
  incoming: NodeActivationLike,
  now?: number,
): NativeNodeActivation {
  assertAvailable()
  return callNative(() => native.mergeActivation(toNativeActivation(existing), toNativeActivation(incoming), now))
}

/**
 * Apply a reinforcement delta to a durable activation record. When `context`
 * is given, the same delta additionally reinforces `activation.context[context]`
 * — an independently-decaying, additional lens, not a replacement for the
 * global score.
 */
export function reinforceActivation(
  previous: NodeActivationLike | undefined,
  delta: number,
  now: number,
  context?: string,
): NativeNodeActivation {
  assertAvailable()
  return callNative(() =>
    native.reinforceActivation(previous ? toNativeActivation(previous) : undefined, delta, now, context)
  )
}

/**
 * Apply a suppression delta to a durable activation record's `inhibition`
 * (mirrors `reinforceActivation` but for the inhibition axis, which decays on
 * its own, shorter-by-default half-life). A negative `delta` releases
 * suppression.
 */
export function suppressActivation(
  previous: NodeActivationLike | undefined,
  delta: number,
  now: number,
): NativeNodeActivation {
  assertAvailable()
  return callNative(() =>
    native.suppressActivation(previous ? toNativeActivation(previous) : undefined, delta, now)
  )
}

/** Current decayed activation score for a stored `score`/anchor pair. */
export function activationScoreOf(
  score: number,
  lastMeaningfulActivation: number,
  now: number,
  halfLifeMs: number,
): number {
  assertAvailable()
  return callNative(() => native.activationScoreOf(score, lastMeaningfulActivation, now, halfLifeMs))
}

/** Conservative token estimate for a serialized memory payload. */
export function estimateNodeTokens(serializedMemory: string): number {
  assertAvailable()
  return callNative(() => native.estimateNodeTokens(serializedMemory))
}

/** Explain a composite activation score from its signal components. */
export function scoreBreakdown(
  semantic: number,
  graph: number,
  recency: number,
  usage: number,
  weights?: { semantic?: number; graph?: number; recency?: number; usage?: number },
): NativeActivationScoreBreakdown {
  assertAvailable()
  return callNative(() => native.scoreBreakdown(
    semantic,
    graph,
    recency,
    usage,
    weights?.semantic,
    weights?.graph,
    weights?.recency,
    weights?.usage,
  ))
}

function assertNonNegativeInteger(n: number, name: string): void {
  if (!Number.isInteger(n) || n < 0) throw new RangeError(`${name} must be a non-negative integer`)
}

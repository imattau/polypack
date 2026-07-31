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
import { cosineSimilarity as cosine } from '../../../src/vector-index.js'
import type { DistanceFunction } from '../../../src/vector-index.js'
import type {
  EngineInfo as BindingEngineInfo,
  NativeBinding,
  NativeExactIndexBinding,
  NativeHnswIndexBinding,
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

function loadNative(): NativeBinding | null {
  try {
    const triple = TRIPLES[`${process.platform}-${process.arch}`]
    if (!triple) return null
    const name = `polypack-native.${triple}.node`
    // Bundled build resolves from dist/; source usage (vitest) resolves from
    // src/ with the binaries in ../dist.
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
    return null
  } catch {
    return null
  }
}

const native: NativeBinding | null = loadNative()

export interface EngineInfo extends BindingEngineInfo {
  available: boolean
}

/** True when the native addon for this platform is loadable. */
export function isNativeAvailable(): boolean {
  return native !== null
}

/**
 * Report which engine is active: `rust-native` when the addon loaded,
 * otherwise the TypeScript fallback.
 */
export function engineInfo(): EngineInfo {
  if (!native) {
    return { graph: 'typescript', vector: 'typescript', storage: 'host', available: false }
  }
  return { ...native.engineInfo(), available: true }
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
 * update-safe HNSW. Only the cosine distance is supported natively; passing a
 * custom distance function throws rather than silently diverging.
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
    if (distanceFn && distanceFn !== cosine) {
      throw new Error('polypack-native HNSW only supports the cosine distance function')
    }
    this.onChange = onChange
    this.inner = new native.NativeHnswIndex(
      {
        m: config?.M,
        mmax0: config?.Mmax0,
        efConstruction: config?.efConstruction,
        efSearch: config?.efSearch,
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

function assertFiniteVector(vector: ArrayLike<number>, name = 'vector'): void {
  if (vector.length === 0) throw new RangeError(`${name} must not be empty`)
  for (let i = 0; i < vector.length; i++) {
    if (!Number.isFinite(vector[i])) throw new RangeError(`${name} must contain finite values`)
  }
}

function assertNonNegativeInteger(n: number, name: string): void {
  if (!Number.isInteger(n) || n < 0) throw new RangeError(`${name} must be a non-negative integer`)
}

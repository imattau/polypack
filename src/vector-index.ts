/** Returns a similarity score where larger values are better matches. */
export type DistanceFunction = (a: ArrayLike<number>, b: ArrayLike<number>) => number
export type VectorPrecision = 'float64' | 'float32'

function storedVector(vector: ArrayLike<number>, precision: VectorPrecision): Float64Array | Float32Array {
  return precision === 'float32' ? new Float32Array(vector) : new Float64Array(vector)
}

function vectorLength(a: ArrayLike<number>, b: ArrayLike<number>): number {
  if (a.length !== b.length) {
    throw new RangeError(`Vector dimension mismatch: ${a.length} !== ${b.length}`)
  }
  return a.length
}

/** Calculate cosine similarity. Throws when vector dimensions differ. */
export function cosineSimilarity(a: ArrayLike<number>, b: ArrayLike<number>): number {
  const len = vectorLength(a, b)
  let dot = 0
  let na = 0
  let nb = 0
  for (let i = 0; i < len; i++) {
    dot += a[i] * b[i]
    na += a[i] * a[i]
    nb += b[i] * b[i]
  }
  const denom = Math.sqrt(na) * Math.sqrt(nb)
  return denom === 0 ? 0 : dot / denom
}

/** Convert Euclidean distance to a 0–1 similarity. Throws on unequal dimensions. */
export function euclideanSimilarity(a: ArrayLike<number>, b: ArrayLike<number>): number {
  const len = vectorLength(a, b)
  let sum = 0
  for (let i = 0; i < len; i++) {
    const diff = a[i] - b[i]
    sum += diff * diff
  }
  return 1 / (1 + Math.sqrt(sum))
}

function vectorNorm(vector: ArrayLike<number>): number {
  let sum = 0
  for (let i = 0; i < vector.length; i++) sum += vector[i] * vector[i]
  return Math.sqrt(sum)
}

function cosineWithNorms(a: ArrayLike<number>, b: ArrayLike<number>, normA: number, normB: number): number {
  const len = vectorLength(a, b)
  const denom = normA * normB
  if (denom === 0) return 0
  let dot = 0
  for (let i = 0; i < len; i++) dot += a[i] * b[i]
  return dot / denom
}

/**
 * Structural surface shared by `VectorIndex`, `HNSWIndex`, and the native
 * engines from `@0xx0lostcause0xx0/polypack-native` (`NativeVectorIndex`,
 * `NativeHnswIndex`). This is what `PolyGraph`'s `createVectorIndex`
 * constructor hook accepts, so any conforming engine can be swapped in
 * without a nominal-typing mismatch against `VectorIndex`'s private fields.
 */
export interface VectorIndexLike {
  add(id: string, vector: number[] | Float64Array | Float32Array): void
  hydrate(id: string, vector: number[] | Float64Array | Float32Array): void
  addMany(entries: Array<{ id: string; vector: number[] | Float64Array | Float32Array }>): void
  remove(id: string): void
  removeMany(ids: string[]): void
  query(vector: ArrayLike<number>, topK: number, threshold?: number): Array<{ id: string; score: number }>
  clear(): void
  readonly size: number
  entries(): IterableIterator<[string, Float64Array]>
  has(id: string): boolean
  get(id: string): Float64Array | undefined
  setPrecision?(precision: VectorPrecision): void
}

/** Exact in-memory vector index with O(n log k) top-k selection. */
export class VectorIndex implements VectorIndexLike {
  private vectors = new Map<string, Float64Array | Float32Array>()
  private norms = new Map<string, number>()
  private onChange?: (id: string) => void
  private distanceFn: DistanceFunction

  constructor(onChange?: (id: string) => void, distanceFn?: DistanceFunction, private precision: VectorPrecision = 'float64') {
    this.onChange = onChange
    this.distanceFn = distanceFn ?? cosineSimilarity
  }

  setPrecision(precision: VectorPrecision): void {
    if (precision === this.precision) return
    this.precision = precision
    for (const [id, vector] of this.vectors) {
      const stored = storedVector(vector, precision)
      this.vectors.set(id, stored)
      if (this.distanceFn === cosineSimilarity) this.norms.set(id, vectorNorm(stored))
    }
  }

  add(id: string, vector: number[] | Float64Array | Float32Array): void {
    if (!id) throw new TypeError('Vector id must not be empty')
    assertFiniteVector(vector)
    const stored = storedVector(vector, this.precision)
    this.vectors.set(id, stored)
    if (this.distanceFn === cosineSimilarity) this.norms.set(id, vectorNorm(stored))
    this.onChange?.(id)
  }

  /** Add an already-persisted vector without marking it dirty again. */
  hydrate(id: string, vector: number[] | Float64Array | Float32Array): void {
    if (!id) throw new TypeError('Vector id must not be empty')
    assertFiniteVector(vector)
    const stored = storedVector(vector, this.precision)
    this.vectors.set(id, stored)
    if (this.distanceFn === cosineSimilarity) this.norms.set(id, vectorNorm(stored))
  }

  addMany(entries: Array<{ id: string; vector: number[] | Float64Array | Float32Array }>): void {
    for (const { id, vector } of entries) {
      if (!id) throw new TypeError('Vector id must not be empty')
      assertFiniteVector(vector)
      const stored = storedVector(vector, this.precision)
      this.vectors.set(id, stored)
      if (this.distanceFn === cosineSimilarity) this.norms.set(id, vectorNorm(stored))
      this.onChange?.(id)
    }
  }

  remove(id: string): void {
    this.vectors.delete(id)
    this.norms.delete(id)
  }

  removeMany(ids: string[]): void {
    for (const id of ids) {
      this.vectors.delete(id)
      this.norms.delete(id)
    }
  }

  query(
    vector: ArrayLike<number>,
    topK: number,
    threshold = 0
  ): Array<{ id: string; score: number }> {
    assertFiniteVector(vector, 'query vector')
    assertNonNegativeInteger(topK, 'topK')
    if (!Number.isFinite(threshold)) throw new RangeError('threshold must be finite')
    if (topK === 0) return []
    const heap: Array<{ id: string; score: number; order: number }> = []
    const queryNorm = this.distanceFn === cosineSimilarity ? vectorNorm(vector) : 0
    let order = 0

    const isLess = (a: typeof heap[number], b: typeof heap[number]) =>
      a.score < b.score || (a.score === b.score && a.order > b.order)
    const siftUp = (index: number) => {
      while (index > 0) {
        const parent = (index - 1) >> 1
        if (!isLess(heap[index], heap[parent])) break
        ;[heap[index], heap[parent]] = [heap[parent], heap[index]]
        index = parent
      }
    }
    const siftDown = (index: number) => {
      while (true) {
        const left = index * 2 + 1
        const right = left + 1
        let smallest = index
        if (left < heap.length && isLess(heap[left], heap[smallest])) smallest = left
        if (right < heap.length && isLess(heap[right], heap[smallest])) smallest = right
        if (smallest === index) break
        ;[heap[index], heap[smallest]] = [heap[smallest], heap[index]]
        index = smallest
      }
    }

    for (const [id, v] of this.vectors) {
      const score = this.distanceFn === cosineSimilarity
        ? cosineWithNorms(vector, v, queryNorm, this.norms.get(id) ?? 0)
        : this.distanceFn(vector, v)
      if (score < threshold) continue
      const candidate = { id, score, order: order++ }
      if (heap.length < topK) {
        heap.push(candidate)
        siftUp(heap.length - 1)
      } else if (score > heap[0].score) {
        heap[0] = candidate
        siftDown(0)
      }
    }
    return heap
      .sort((a, b) => b.score - a.score || a.order - b.order)
      .map(({ id, score }) => ({ id, score }))
  }

  clear(): void {
    this.vectors.clear()
    this.norms.clear()
  }

  get size(): number {
    return this.vectors.size
  }

  *entries(): IterableIterator<[string, Float64Array]> {
    for (const [id, vector] of this.vectors) yield [id, new Float64Array(vector)]
  }

  has(id: string): boolean {
    return this.vectors.has(id)
  }

  get(id: string): Float64Array | undefined {
    const vector = this.vectors.get(id)
    return vector ? new Float64Array(vector) : undefined
  }
}
import { assertFiniteVector, assertNonNegativeInteger } from './utils.js'

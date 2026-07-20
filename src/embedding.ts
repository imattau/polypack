import { assertFiniteVector } from './utils.js'

/** Vector values accepted from an embedding provider. */
export type EmbeddingVector = number[] | Float32Array | Float64Array

/** Pluggable text embedding contract for local models, APIs, or custom logic. */
export interface EmbeddingProvider {
  readonly dimensions?: number
  embed(text: string): EmbeddingVector | Promise<EmbeddingVector>
}

/** Configuration for the dependency-free feature-hashing provider. */
export interface FeatureHashEmbeddingOptions {
  dimensions?: number
}

/**
 * Dependency-free lexical embedding based on hashed word frequencies.
 * It is deterministic, normalized for cosine similarity, and requires no model
 * download. Applications needing semantic meaning can supply another provider.
 */
export class FeatureHashEmbedding implements EmbeddingProvider {
  readonly dimensions: number

  constructor(options: FeatureHashEmbeddingOptions = {}) {
    const dimensions = options.dimensions ?? 384
    if (!Number.isInteger(dimensions) || dimensions <= 0) {
      throw new RangeError('Embedding dimensions must be a positive integer')
    }
    this.dimensions = dimensions
  }

  embed(text: string): Float64Array {
    if (typeof text !== 'string') throw new TypeError('Embedding text must be a string')
    const vector = new Float64Array(this.dimensions)
    const words = text.toLowerCase().split(/[^\p{L}\p{N}]+/u).filter(Boolean)

    for (const word of words) {
      let first = 0x811c9dc5
      let second = 0x6b8b4567
      for (let index = 0; index < word.length; index++) {
        const code = word.charCodeAt(index)
        first = Math.imul(first ^ code, 0x01000193)
        second = Math.imul(second ^ code, 0x5bd1e995)
      }
      vector[((first ^ second) >>> 0) % this.dimensions]++
    }

    let squaredNorm = 0
    for (const value of vector) squaredNorm += value * value
    const norm = Math.sqrt(squaredNorm)
    if (norm > 0) {
      for (let index = 0; index < vector.length; index++) vector[index] /= norm
    }
    return vector
  }
}

/** Validate and detach a provider result for safe graph ownership. */
export async function createEmbedding(provider: EmbeddingProvider, text: string): Promise<Float64Array> {
  if (typeof text !== 'string') throw new TypeError('Embedding text must be a string')
  const result = await provider.embed(text)
  if (!Array.isArray(result) && !(result instanceof Float32Array) && !(result instanceof Float64Array)) {
    throw new TypeError('Embedding provider must return a number array or typed array')
  }
  const vector = new Float64Array(result)
  if (vector.length === 0) throw new RangeError('Embedding provider must return at least one dimension')
  assertFiniteVector(vector, 'embedding')
  if (provider.dimensions !== undefined && vector.length !== provider.dimensions) {
    throw new RangeError(`Embedding provider returned ${vector.length} dimensions; expected ${provider.dimensions}`)
  }
  return vector
}

/**
 * Build a weighted embedding text by repeating fields according to their weight.
 * The default weight is 1; fields with weight 3 are repeated 3× so the
 * feature-hash bag-of-words embedding treats them as more significant.
 *
 * ```ts
 * buildEmbeddingText({ subject: 'Hello', content: 'World' }, { subject: 3 })
 * // => "Hello Hello Hello World"
 * ```
 */
export function buildEmbeddingText(
  fields: Record<string, string>,
  weights?: Record<string, number>,
): string {
  const parts: string[] = []
  for (const [key, value] of Object.entries(fields)) {
    if (!value) continue
    const w = weights?.[key] ?? 1
    for (let i = 0; i < w; i++) parts.push(value)
  }
  return parts.join(' ')
}

/** Default model-free 384-dimensional embedding provider. */
export const defaultEmbedding = new FeatureHashEmbedding()

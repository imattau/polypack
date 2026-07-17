import { describe, it, expect } from 'vitest'
import { VectorIndex, cosineSimilarity, euclideanSimilarity } from '../src/vector-index'
import type { DistanceFunction } from '../src/vector-index'

describe('cosineSimilarity', () => {
  it('returns 1 for identical vectors', () => {
    expect(cosineSimilarity([1, 2, 3], [1, 2, 3])).toBeCloseTo(1)
  })

  it('returns 0 for orthogonal vectors', () => {
    expect(cosineSimilarity([1, 0, 0], [0, 1, 0])).toBeCloseTo(0)
  })

  it('returns -1 for opposite vectors', () => {
    expect(cosineSimilarity([1, 0], [-1, 0])).toBeCloseTo(-1)
  })

  it('returns 0 when one vector is all zeros', () => {
    expect(cosineSimilarity([0, 0, 0], [1, 2, 3])).toBe(0)
  })
})

describe('VectorIndex', () => {
  it('adds and retrieves vectors', () => {
    const idx = new VectorIndex()
    idx.add('a', [1, 0, 0])
    expect(idx.has('a')).toBe(true)
    expect(idx.get('a')).toEqual([1, 0, 0])
  })

  it('queries top-K similar vectors', () => {
    const idx = new VectorIndex()
    idx.add('v1', [1, 0, 0])
    idx.add('v2', [0.9, 0.1, 0])
    idx.add('v3', [0, 1, 0])
    idx.add('v4', [0, 0, 1])

    const results = idx.query([1, 0, 0], 3)
    expect(results).toHaveLength(3)
    expect(results[0].id).toBe('v1') // most similar
    expect(results[1].id).toBe('v2')
  })

  it('respects threshold', () => {
    const idx = new VectorIndex()
    idx.add('a', [1, 0, 0])
    idx.add('b', [0.5, 0.5, 0])
    idx.add('c', [0, 1, 0])

    const results = idx.query([1, 0, 0], 10, 0.8)
    expect(results).toHaveLength(1)
    expect(results[0].id).toBe('a')
  })

  it('removes vectors', () => {
    const idx = new VectorIndex()
    idx.add('a', [1, 0, 0])
    idx.remove('a')
    expect(idx.has('a')).toBe(false)
    expect(idx.size).toBe(0)
  })

  it('clears all vectors', () => {
    const idx = new VectorIndex()
    idx.add('a', [1, 0, 0])
    idx.add('b', [0, 1, 0])
    idx.clear()
    expect(idx.size).toBe(0)
  })

  it('calls onChange callback on add', () => {
    const changed: string[] = []
    const idx = new VectorIndex((id) => changed.push(id))
    idx.add('x', [1, 0, 0])
    expect(changed).toEqual(['x'])
  })

  it('addMany adds multiple vectors at once', () => {
    const idx = new VectorIndex()
    idx.addMany([
      { id: 'a', vector: [1, 0] },
      { id: 'b', vector: [0, 1] },
    ])
    expect(idx.size).toBe(2)
    expect(idx.has('a')).toBe(true)
    expect(idx.has('b')).toBe(true)
  })

  it('removeMany removes multiple vectors at once', () => {
    const idx = new VectorIndex()
    idx.add('a', [1, 0])
    idx.add('b', [0, 1])
    idx.add('c', [1, 1])
    idx.removeMany(['a', 'c'])
    expect(idx.size).toBe(1)
    expect(idx.has('b')).toBe(true)
  })

  it('supports custom distance function', () => {
    const manhattan: DistanceFunction = (a, b) => {
      let sum = 0
      for (let i = 0; i < a.length; i++) sum += Math.abs(a[i] - b[i])
      return 1 / (1 + sum)
    }
    const idx = new VectorIndex(undefined, manhattan)
    idx.add('a', [1, 0])
    idx.add('b', [0, 1])
    idx.add('c', [0.9, 0.1])

    const results = idx.query([1, 0], 2)
    expect(results[0].id).toBe('a')
    expect(results[1].id).toBe('c')
  })
})

describe('VectorIndex edge cases', () => {
  it('query on empty index returns empty array', () => {
    const idx = new VectorIndex()
    expect(idx.query([1, 0, 0], 5)).toHaveLength(0)
  })

  it('query with topK > size returns all', () => {
    const idx = new VectorIndex()
    idx.add('a', [1, 0])
    idx.add('b', [0, 1])
    const results = idx.query([1, 0], 100)
    expect(results).toHaveLength(2)
  })

  it('query with threshold > 1 returns empty (cosine max is 1)', () => {
    const idx = new VectorIndex()
    idx.add('a', [1, 0])
    const results = idx.query([1, 0], 5, 1.1)
    expect(results).toHaveLength(0)
  })

  it('entries iterates all stored pairs', () => {
    const idx = new VectorIndex()
    idx.add('a', [1, 0])
    idx.add('b', [0, 1])
    const entries = [...idx.entries()]
    expect(entries).toHaveLength(2)
  })

  it('addMany with empty array is a no-op', () => {
    const idx = new VectorIndex()
    idx.addMany([])
    expect(idx.size).toBe(0)
  })

  it('removeMany with empty array is a no-op', () => {
    const idx = new VectorIndex()
    idx.add('a', [1, 0])
    idx.removeMany([])
    expect(idx.size).toBe(1)
  })
})

describe('cosineSimilarity edge cases', () => {
  it('handles vectors of different lengths (clamps to shorter)', () => {
    const score = cosineSimilarity([1, 0, 0], [1, 0])
    expect(score).toBeCloseTo(1) // only first 2 dims compared
  })

  it('handles both vectors all-zeros', () => {
    expect(cosineSimilarity([0, 0], [0, 0])).toBe(0)
  })

  it('is not NaN for different lengths', () => {
    const score = cosineSimilarity([1, 2, 3], [1])
    expect(Number.isNaN(score)).toBe(false)
  })
})

describe('euclideanSimilarity edge cases', () => {
  it('handles vectors of different lengths', () => {
    const score = euclideanSimilarity([3, 4], [3, 4, 5])
    expect(Number.isNaN(score)).toBe(false)
    expect(score).toBeCloseTo(1 / (1 + 0)) // first 2 dims match
  })
})

describe('euclideanSimilarity', () => {
  it('returns 1 for identical vectors', () => {
    expect(euclideanSimilarity([3, 4], [3, 4])).toBeCloseTo(1)
  })

  it('returns smaller values for distant vectors', () => {
    const close = euclideanSimilarity([1, 0], [1.1, 0])
    const far = euclideanSimilarity([1, 0], [100, 0])
    expect(close).toBeGreaterThan(far)
  })
})

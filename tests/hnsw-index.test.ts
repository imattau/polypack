import { describe, it, expect } from 'vitest'
import { HNSWIndex } from '../src/hnsw-index'
import { VectorIndex, cosineSimilarity, euclideanSimilarity } from '../src/vector-index'

describe('HNSWIndex', () => {
  describe('CRUD', () => {
    it('adds and retrieves a vector', () => {
      const index = new HNSWIndex()
      index.add('v1', [1, 2, 3])
      expect(index.has('v1')).toBe(true)
      expect([...index.get('v1')!]).toEqual([1, 2, 3])
    })

    it('returns undefined for missing vector', () => {
      const index = new HNSWIndex()
      expect(index.get('missing')).toBeUndefined()
      expect(index.has('missing')).toBe(false)
    })

    it('shares the internal Float64Array reference', () => {
      const index = new HNSWIndex()
      const vec = [1, 2, 3]
      index.add('v1', vec)
      const got = index.get('v1')
      got![0] = 999
      expect([...index.get('v1')!]).toEqual([999, 2, 3])
    })

    it('removes a vector', () => {
      const index = new HNSWIndex()
      index.add('v1', [1, 2, 3])
      index.remove('v1')
      expect(index.has('v1')).toBe(false)
      expect(index.get('v1')).toBeUndefined()
    })

    it('remove is idempotent', () => {
      const index = new HNSWIndex()
      index.remove('never-added')
      // no throw
    })

    it('removeMany removes all specified ids', () => {
      const index = new HNSWIndex()
      index.add('a', [1, 0, 0])
      index.add('b', [0, 1, 0])
      index.add('c', [0, 0, 1])
      index.removeMany(['a', 'c'])
      expect(index.has('a')).toBe(false)
      expect(index.has('b')).toBe(true)
      expect(index.has('c')).toBe(false)
    })

    it('clears all vectors', () => {
      const index = new HNSWIndex()
      index.add('v1', [1, 0, 0])
      index.add('v2', [0, 1, 0])
      index.clear()
      expect(index.size).toBe(0)
      expect(index.has('v1')).toBe(false)
      expect(index.get('v2')).toBeUndefined()
    })

    it('size reflects stored vector count', () => {
      const index = new HNSWIndex()
      expect(index.size).toBe(0)
      index.add('v1', [1, 0, 0])
      expect(index.size).toBe(1)
      index.add('v2', [0, 1, 0])
      expect(index.size).toBe(2)
      index.remove('v1')
      expect(index.size).toBe(1)
      index.clear()
      expect(index.size).toBe(0)
    })

    it('throws on empty id', () => {
      const index = new HNSWIndex()
      expect(() => index.add('', [1, 2, 3])).toThrow()
    })

    it('throws on non-finite vector values', () => {
      const index = new HNSWIndex()
      expect(() => index.add('v1', [1, NaN, 3])).toThrow()
      expect(() => index.add('v1', [1, Infinity, 3])).toThrow()
    })

    it('hydrate adds a vector without triggering onChange', () => {
      let changed = false
      const index = new HNSWIndex(() => { changed = true })
      index.hydrate('v1', [1, 2, 3])
      expect(changed).toBe(false)
      expect(index.has('v1')).toBe(true)
    })

    it('add triggers onChange', () => {
      let changedId = ''
      const index = new HNSWIndex((id) => { changedId = id })
      index.add('v1', [1, 2, 3])
      expect(changedId).toBe('v1')
    })

    it('addMany adds all and triggers onChange per entry', () => {
      const changed: string[] = []
      const index = new HNSWIndex((id) => { changed.push(id) })
      index.addMany([
        { id: 'a', vector: [1, 0, 0] },
        { id: 'b', vector: [0, 1, 0] },
      ])
      expect(index.size).toBe(2)
      expect(changed).toEqual(['a', 'b'])
    })

    it('entries iterates stored vectors', () => {
      const index = new HNSWIndex()
      index.add('a', [1, 0])
      index.add('b', [0, 1])
      const entries = [...index.entries()]
      expect(entries).toHaveLength(2)
      const sorted = [...entries].sort(([a], [b]) => a.localeCompare(b))
      expect([...sorted[0][1]]).toEqual([1, 0])
      expect([...sorted[1][1]]).toEqual([0, 1])
    })

    it('entries skips removed vectors', () => {
      const index = new HNSWIndex()
      index.add('a', [1, 0])
      index.add('b', [0, 1])
      index.remove('a')
      const entries = [...index.entries()]
      expect(entries).toHaveLength(1)
      expect(entries[0][0]).toBe('b')
    })
  })

  describe('query', () => {
    function bruteForce(
      index: VectorIndex,
      query: number[],
      topK: number,
      threshold: number,
    ): string[] {
      return index.query(query, topK, threshold).map(r => r.id)
    }

    it('returns empty for empty index', () => {
      const index = new HNSWIndex()
      expect(index.query([1, 0, 0], 5)).toEqual([])
    })

    it('returns empty when topK is 0', () => {
      const index = new HNSWIndex()
      index.add('v1', [1, 0, 0])
      expect(index.query([1, 0, 0], 0)).toEqual([])
    })

    it('finds nearest neighbors', () => {
      const index = new HNSWIndex()
      index.add('a', [1, 0, 0])
      index.add('b', [0, 1, 0])
      index.add('c', [0, 0, 1])

      const results = index.query([1, 0.1, 0], 2, 0)
      expect(results).toHaveLength(2)
      expect(results[0].id).toBe('a')
      expect(results[0].score).toBeGreaterThan(results[1].score)
    })

    it('filters by threshold', () => {
      const index = new HNSWIndex()
      index.add('a', [1.0, 0, 0])
      index.add('b', [0.5, 0.5, 0])
      index.add('c', [0, 0, 1])

      const results = index.query([1, 0, 0], 5, 0.9)
      expect(results).toHaveLength(1)
      expect(results[0].id).toBe('a')
    })

    it('matches brute-force for a small dataset', () => {
      const dims = 8
      const count = 200
      const vectors: number[][] = []
      for (let i = 0; i < count; i++) {
        const v: number[] = []
        for (let d = 0; d < dims; d++) v.push(Math.random() * 2 - 1)
        vectors.push(v)
      }

      const exact = new VectorIndex()
      const ann = new HNSWIndex(undefined, undefined, { M: 16, efConstruction: 200, efSearch: 200 })

      for (let i = 0; i < count; i++) {
        exact.add(`v${i}`, vectors[i])
        ann.add(`v${i}`, vectors[i])
      }

      for (let iter = 0; iter < 10; iter++) {
        const q: number[] = []
        for (let d = 0; d < dims; d++) q.push(Math.random() * 2 - 1)

        const exactIds = new Set(bruteForce(exact, q, 10, 0))
        const annResults = ann.query(q, 10, 0)

        // HNSW should find at least 90% recall (top-10)
        const matched = annResults.filter(r => exactIds.has(r.id)).length
        expect(matched).toBeGreaterThanOrEqual(9)
      }
    })

    it('works with euclidean distance', () => {
      const index = new HNSWIndex(undefined, euclideanSimilarity)
      index.add('a', [1, 0])
      index.add('b', [0, 1])
      index.add('c', [0, 0])

      const results = index.query([0.9, 0], 2, 0)
      expect(results[0].id).toBe('a')
    })

    it('scores match brute-force scores for the top result', () => {
      const index = new HNSWIndex()
      index.add('a', [1, 0, 0])
      index.add('b', [0, 1, 0])

      const exact = new VectorIndex()
      exact.add('a', [1, 0, 0])
      exact.add('b', [0, 1, 0])

      const annRes = index.query([1, 0.01, 0], 2, 0)
      const exactRes = exact.query([1, 0.01, 0], 2, 0)

      expect(annRes[0].score).toBeCloseTo(exactRes[0].score, 6)
    })

    it('rejects non-finite query vectors', () => {
      const index = new HNSWIndex()
      expect(() => index.query([NaN, 0], 5)).toThrow()
      expect(() => index.query([Infinity, 0], 5)).toThrow()
    })

    it('rejects non-integer topK', () => {
      const index = new HNSWIndex()
      expect(() => index.query([1, 0], 1.5)).toThrow()
    })

    it('rejects infinite threshold', () => {
      const index = new HNSWIndex()
      expect(() => index.query([1, 0], 5, Infinity)).toThrow()
    })
  })

  describe('recall at scale', () => {
    it('achieves >95% recall@10 with 1000 vectors (8 dims)', () => {
      const dims = 8
      const count = 1000
      const vectors: number[][] = []
      for (let i = 0; i < count; i++) {
        const v: number[] = []
        for (let d = 0; d < dims; d++) v.push(Math.random() * 2 - 1)
        vectors.push(v)
      }

      const exact = new VectorIndex()
      const ann = new HNSWIndex(undefined, undefined, {
        M: 16,
        efConstruction: 200,
        efSearch: 200,
      })

      for (let i = 0; i < count; i++) {
        exact.add(`v${i}`, vectors[i])
        ann.add(`v${i}`, vectors[i])
      }

      let totalRecall = 0
      const trials = 20

      for (let iter = 0; iter < trials; iter++) {
        const q: number[] = []
        for (let d = 0; d < dims; d++) q.push(Math.random() * 2 - 1)

        const exactIds = new Set(exact.query(q, 10, 0).map(r => r.id))
        const annIds = new Set(ann.query(q, 10, 0).map(r => r.id))

        const intersection = [...exactIds].filter(id => annIds.has(id)).length
        totalRecall += intersection / exactIds.size
      }

      const avgRecall = totalRecall / trials
      console.log(`\n  HNSW recall@10 (1K vectors, 8 dims, 20 trials): ${(avgRecall * 100).toFixed(1)}%`)
      expect(avgRecall).toBeGreaterThanOrEqual(0.95)
    })
  })

  describe('distance function integration', () => {
    it('accepts a custom distance function', () => {
      const customDist: typeof cosineSimilarity = (a, b) => {
        let sum = 0
        for (let i = 0; i < a.length; i++) sum += Math.abs(a[i] - b[i])
        return 1 / (1 + sum)
      }

      const index = new HNSWIndex(undefined, customDist)
      index.add('a', [0, 0])
      index.add('b', [1, 1])

      const results = index.query([0, 0], 2, 0)
      expect(results[0].id).toBe('a')
    })
  })
})

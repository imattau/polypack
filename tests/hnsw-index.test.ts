import { describe, it, expect } from 'vitest'
import { HNSWIndex } from '../src/hnsw-index'
import { VectorIndex, cosineSimilarity, euclideanSimilarity } from '../src/vector-index'
import type { VectorIndexLike } from '../src/vector-index'

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

    it('returns detached copies from get and entries', () => {
      const index = new HNSWIndex()
      const vec = [1, 2, 3]
      index.add('v1', vec)
      const got = index.get('v1')!
      got[0] = 999
      expect([...index.get('v1')!]).toEqual([1, 2, 3])
      vec[0] = 42
      expect([...index.get('v1')!]).toEqual([1, 2, 3])
      const [entryId, entryVec] = [...index.entries()][0]
      expect(entryId).toBe('v1')
      entryVec[1] = 123
      expect([...index.get('v1')!]).toEqual([1, 2, 3])
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
      index: VectorIndexLike,
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

  describe('ID reuse and replacement', () => {
    it('re-adds a removed id and serves it from query and entries', () => {
      const index = new HNSWIndex()
      index.add('a', [1, 0, 0])
      index.add('b', [0, 1, 0])
      index.remove('a')
      expect(index.size).toBe(1)
      expect(index.has('a')).toBe(false)
      index.add('a', [1, 0, 0])
      expect(index.size).toBe(2)
      expect(index.has('a')).toBe(true)
      expect([...index.entries()].map(([id]) => id).sort()).toEqual(['a', 'b'])
      const results = index.query([1, 0, 0], 5)
      expect(results[0].id).toBe('a')
    })

    it('update replaces a vector without stale topology', () => {
      const index = new HNSWIndex(undefined, undefined, { M: 16, efConstruction: 200, efSearch: 200 })
      index.add('a', [1, 0, 0])
      index.add('b', [0, 1, 0])
      index.add('c', [0, 0, 1])
      index.add('d', [0.5, 0.5, 0])
      index.update('b', [1, 0, 0])
      const results = index.query([0, 1, 0], 5)
      expect(results[0].id).not.toBe('b')
      expect(index.get('b')).toBeDefined()
      expect(index.size).toBe(4)
    })

    it('update does not duplicate an id', () => {
      const index = new HNSWIndex()
      index.add('a', [1, 0])
      index.add('b', [0, 1])
      index.update('a', [1, 0])
      expect(index.size).toBe(2)
      expect([...index.entries()].map(([id]) => id).sort()).toEqual(['a', 'b'])
    })

    it('keeps recall after remove and re-add churn', () => {
      const dims = 8
      const count = 500
      const rand = mulberry32(1234)
      const exact = new VectorIndex()
      // Level assignment must be deterministic too, so pass a seeded RNG.
      const ann = new HNSWIndex(undefined, undefined, { M: 16, efConstruction: 200, efSearch: 200 }, mulberry32(999))
      const vecs: number[][] = []
      for (let i = 0; i < count; i++) {
        const v = Array.from({ length: dims }, () => rand() * 2 - 1)
        vecs.push(v)
        exact.add(`v${i}`, v)
        ann.add(`v${i}`, v)
      }

      // Remove half, then re-add them with new vectors.
      for (let i = 0; i < count; i += 2) {
        exact.remove(`v${i}`)
        ann.remove(`v${i}`)
      }
      for (let i = 0; i < count; i += 2) {
        const v = Array.from({ length: dims }, () => rand() * 2 - 1)
        vecs[i] = v
        exact.add(`v${i}`, v)
        ann.add(`v${i}`, v)
      }
      // Update a quarter of the remainder in place.
      for (let i = 1; i < count; i += 4) {
        const v = Array.from({ length: dims }, () => rand() * 2 - 1)
        vecs[i] = v
        exact.add(`v${i}`, v)
        ann.update(`v${i}`, v)
      }
      void vecs
      expect(ann.size).toBe(exact.size)

      let totalRecall = 0
      const trials = 20
      for (let t = 0; t < trials; t++) {
        const q = Array.from({ length: dims }, () => rand() * 2 - 1)
        const exactIds = new Set(exact.query(q, 10, 0).map(r => r.id))
        const annIds = new Set(ann.query(q, 10, 0).map(r => r.id))
        totalRecall += [...exactIds].filter(id => annIds.has(id)).length / exactIds.size
      }
      expect(totalRecall / trials).toBeGreaterThanOrEqual(0.9)
    })

    it('clears topology after removing every node', () => {
      const index = new HNSWIndex()
      for (let i = 0; i < 200; i++) {
        index.add(`n${i}`, [Math.random(), Math.random()])
      }
      for (let i = 0; i < 200; i++) {
        index.remove(`n${i}`)
      }
      expect(index.size).toBe(0)
      index.add('fresh', [1, 0])
      const results = index.query([1, 0], 5)
      expect(results[0].id).toBe('fresh')
    })
  })
})

function mulberry32(seed: number): () => number {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

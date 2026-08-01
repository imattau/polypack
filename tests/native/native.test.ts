import { describe, it, expect, beforeAll } from 'vitest'
import {
  NativeVectorIndex,
  NativeHnswIndex,
  engineInfo,
  isNativeAvailable,
  createNativeVectorIndex,
  mergeActivation,
  decayFactor,
  reinforceActivation,
  activationScoreOf,
} from '../../packages/node-native/src/index'
import { PolyGraph, VectorIndex } from '../../src/index'
import { loadFixtures, runFixture } from '../conformance/runner'

const available = isNativeAvailable()

beforeAll(() => {
  if (!available) console.warn('native binary unavailable — skipping native tests')
})

describe('native engine', () => {
  it('reports rust-native when the addon is available', () => {
    if (!available) return
    const info = engineInfo()
    expect(info.available).toBe(true)
    expect(info.vector).toBe('rust-native')
  })
})

describe('NativeVectorIndex parity', () => {
  it('matches the TypeScript VectorIndex results', () => {
    if (!available) return
    const native = new NativeVectorIndex()
    const ts = new VectorIndex()
    const rows = [
      ['a', [1, 0, 0, 0]],
      ['b', [0.8, 0.6, 0, 0]],
      ['c', [0.6, 0.8, 0, 0]],
      ['d', [0, 1, 0, 0]],
      ['e', [0, 0, 1, 0]],
    ] as const
    for (const [id, v] of rows) {
      native.add(id, [...v])
      ts.add(id, [...v])
    }
    expect(native.size).toBe(ts.size)
    const q = [1, 0, 0, 0]
    const nativeIds = native.query(q, 3).map(r => r.id)
    const tsIds = ts.query(q, 3).map(r => r.id)
    expect(nativeIds).toEqual(tsIds)
    expect([...native.entries()].map(([id]) => id).sort()).toEqual([...ts.entries()].map(([id]) => id).sort())
    expect([...native.get('a')!]).toEqual([...ts.get('a')!])
  })

  it('validates inputs like the TypeScript implementation', () => {
    if (!available) return
    const idx = new NativeVectorIndex()
    expect(() => idx.add('', [1, 0])).toThrow(TypeError)
    expect(() => idx.add('a', [1, Number.NaN])).toThrow(RangeError)
    expect(() => idx.query([1, 0], 0)).not.toThrow()
    expect(idx.query([1, 0], 0)).toEqual([])
  })
})

describe('NativeHnswIndex', () => {
  it('matches exact top-k on a seeded dataset', () => {
    if (!available) return
    const hnsw = new NativeHnswIndex(undefined, undefined, { M: 16, efConstruction: 200, efSearch: 300 })
    const exact = new VectorIndex()
    const rand = mulberry32(42)
    for (let i = 0; i < 500; i++) {
      const v = Array.from({ length: 8 }, () => rand() * 2 - 1)
      hnsw.add(`v${i}`, v)
      exact.add(`v${i}`, v)
    }
    let hits = 0
    for (let t = 0; t < 20; t++) {
      const q = Array.from({ length: 8 }, () => rand() * 2 - 1)
      const exactIds = new Set(exact.query(q, 10).map(r => r.id))
      hits += hnsw.query(q, 10).filter(r => exactIds.has(r.id)).length
    }
    expect(hits / (20 * 10)).toBeGreaterThanOrEqual(0.95)
  })

  it('handles remove, re-add, and update churn', () => {
    if (!available) return
    const hnsw = new NativeHnswIndex(undefined, undefined, { M: 16, efConstruction: 200, efSearch: 300 })
    hnsw.add('a', [1, 0, 0, 0, 0, 0, 0, 0])
    hnsw.add('b', [0, 1, 0, 0, 0, 0, 0, 0])
    hnsw.remove('a')
    expect(hnsw.size).toBe(1)
    expect(hnsw.has('a')).toBe(false)
    hnsw.add('a', [1, 0, 0, 0, 0, 0, 0, 0])
    hnsw.update('b', [1, 0, 0, 0, 0, 0, 0, 0])
    expect(hnsw.size).toBe(2)
    // a and b are now identical vectors; both must be returned, order is a tie.
    const results = hnsw.query([1, 0, 0, 0, 0, 0, 0, 0], 2)
    expect(results.map(r => r.id).sort()).toEqual(['a', 'b'])
  })
})

describe('PolyGraph with native vector index', () => {
  it('runs the graph with createNativeVectorIndex', () => {
    if (!available) return
    const graph = new PolyGraph(undefined, undefined, undefined, undefined, createNativeVectorIndex())
    graph.addNode({
      id: 'n1',
      type: 'doc',
      data: { title: 'one' },
      vector: new Float64Array([1, 0, 0]),
      insertedAt: 1,
      updatedAt: 1,
    })
    graph.addNode({
      id: 'n2',
      type: 'doc',
      data: { title: 'two' },
      vector: new Float64Array([0.8, 0.6, 0]),
      insertedAt: 2,
      updatedAt: 2,
    })
    const top = graph.vectors.query([1, 0, 0], 1).map(r => r.id)
    expect(top).toEqual(['n1'])
    expect(graph.vectors.size).toBe(2)
    graph.clear()
  })

  it('passes the conformance graph fixtures against the native index', () => {
    if (!available) return
    for (const fixture of loadFixtures()) {
      expect(() => runFixture(fixture, createNativeVectorIndex())).not.toThrow()
    }
  })
})

describe('native activation helpers', () => {
  it('computes decay and merge matching the TypeScript semantics', () => {
    if (!available) return
    const DAY = 86_400_000
    expect(decayFactor(DAY, DAY)).toBeCloseTo(0.5, 10)
    expect(decayFactor(2 * DAY, DAY)).toBeCloseTo(0.25, 10)
    expect(decayFactor(DAY, 0)).toBe(1)

    const merged = mergeActivation(
      { score: 0.6, importance: 0.3, reinforcementCount: 2, lastMeaningfulActivation: 0 },
      { score: 0.9, importance: 0.1, reinforcementCount: 1, lastMeaningfulActivation: 0 },
      0,
    )
    expect(merged.score).toBeCloseTo(0.9, 10)
    expect(merged.importance).toBeCloseTo(0.3, 10)
    expect(merged.reinforcementCount).toBe(2)

    const reinforced = reinforceActivation(undefined, 0.5, 1000)
    expect(reinforced.score).toBeCloseTo(0.5, 10)
    expect(reinforced.importance).toBeCloseTo(0.025, 10)
    expect(reinforced.reinforcementCount).toBe(1)
    expect(reinforced.lastMeaningfulActivation).toBe(1000)

    expect(activationScoreOf(1, 0, DAY, DAY)).toBeCloseTo(0.5, 10)
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

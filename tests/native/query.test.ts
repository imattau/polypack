import { describe, it, expect, beforeAll } from 'vitest'
import {
  installNativeQueryExecutor,
  isNativeAvailable,
  engineInfo,
} from '../../packages/node-native/src/index'
import { setNativeQueryExecutor } from '../../src/query'
import { PolyGraph } from '../../src/index'
import { loadFixtures, runFixture } from '../conformance/runner'

const available = isNativeAvailable()

beforeAll(() => {
  if (!available) console.warn('native binary unavailable — skipping native query tests')
  installNativeQueryExecutor()
})

function seedGraph(g: PolyGraph): void {
  for (const [id, cat, score] of [
    ['d1', 'science', 0.9],
    ['d2', 'science', 0.5],
    ['d3', 'food', 0.8],
    ['d4', 'science', 0.7],
  ] as const) {
    g.addNode({
      id,
      type: 'document',
      data: { category: cat, score },
      vector: new Float64Array([score, 1 - score]),
      insertedAt: 1,
      updatedAt: 1,
    })
  }
  g.addEdge('d1', 'CITES', 'd2')
  g.addEdge('d4', 'CITES', 'd2')
}

describe('native query executor', () => {
  it('reports the rust-native query engine when installed', () => {
    if (!available) return
    expect(engineInfo().query).toBe('rust-native')
  })

  it('returns the same results as the TypeScript pipeline', () => {
    if (!available) return
    const g = new PolyGraph()
    seedGraph(g)

    setNativeQueryExecutor(null)
    const tsIds = g.query().whereNodeType('document').where('category', 'science').orderBy('score', 'desc').ids()
    const tsCount = g.query().whereNodeType('document').count()

    installNativeQueryExecutor()
    const nativeIds = g.query().whereNodeType('document').where('category', 'science').orderBy('score', 'desc').ids()
    const nativeCount = g.query().whereNodeType('document').count()

    expect(nativeIds).toEqual(tsIds)
    expect(nativeCount).toBe(tsCount)
  })

  it('delegates traversal and similarity to the executor', () => {
    if (!available) return
    const g = new PolyGraph()
    seedGraph(g)

    setNativeQueryExecutor(null)
    const tsTraversal = g.query().where('category', 'science').traverse('CITES', 1, 'out').ids()
    const tsSimilar = g.query().whereNodeType('document').similarTo([1, 0], 0.5, 2).ids()

    installNativeQueryExecutor()
    const nativeTraversal = g.query().where('category', 'science').traverse('CITES', 1, 'out').ids()
    const nativeSimilar = g.query().whereNodeType('document').similarTo([1, 0], 0.5, 2).ids()

    expect(nativeTraversal).toEqual(tsTraversal)
    expect(nativeSimilar).toEqual(tsSimilar)
  })

  it('falls back to TypeScript for join predicates', () => {
    if (!available) return
    const g = new PolyGraph()
    seedGraph(g)
    // join with a predicate is not IR-expressible; both paths agree.
    const viaNative = g
      .query()
      .whereNodeType('document')
      .join('CITES', 'out', n => (n.data as { category: string }).category === 'science')
      .ids()
    setNativeQueryExecutor(null)
    const viaTs = g
      .query()
      .whereNodeType('document')
      .join('CITES', 'out', n => (n.data as { category: string }).category === 'science')
      .ids()
    installNativeQueryExecutor()
    expect(viaNative).toEqual(viaTs)
  })

  it('passes the conformance graph fixtures with the executor installed', () => {
    if (!available) return
    for (const fixture of loadFixtures()) {
      expect(() => runFixture(fixture)).not.toThrow()
    }
  })
})

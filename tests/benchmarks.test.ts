import { describe, it, expect } from 'vitest'
import { PolyGraph } from '../src/graph'
import { MemoryAdapter } from '../src/persistence/memory'

const COUNT = 10_000

function formatMs(ms: number): string {
  return ms < 1 ? `${(ms * 1000).toFixed(0)}µs` : `${ms.toFixed(2)}ms`
}

describe('Performance benchmarks', () => {
  it(`inserts ${COUNT} nodes in under 500ms`, () => {
    const graph = new PolyGraph()
    const t0 = performance.now()
    for (let i = 0; i < COUNT; i++) {
      graph.addNode({
        id: `n${i}`,
        type: ['user', 'post', 'comment'][i % 3],
        data: { idx: i, value: Math.random(), tag: `tag_${i % 50}` },
        insertedAt: i,
        updatedAt: i,
      })
    }
    const t1 = performance.now()
    const elapsed = t1 - t0
    console.log(`  Insert ${COUNT} nodes: ${formatMs(elapsed)} (${(COUNT / (elapsed / 1000)).toFixed(0)} nodes/sec)`)
    expect(graph.size).toBe(COUNT)
    expect(elapsed).toBeLessThan(500)
  })

  it(`inserts ${COUNT} edges in under 500ms`, () => {
    const graph = new PolyGraph()
    for (let i = 0; i < COUNT; i++) {
      graph.addNode({ id: `n${i}`, type: 't', data: {}, insertedAt: i, updatedAt: i })
    }
    const t0 = performance.now()
    for (let i = 1; i < COUNT; i++) {
      graph.addEdge(`n${i - 1}`, 'NEXT', `n${i}`)
    }
    const t1 = performance.now()
    const elapsed = t1 - t0
    console.log(`  Insert ${COUNT - 1} edges: ${formatMs(elapsed)} (${((COUNT - 1) / (elapsed / 1000)).toFixed(0)} edges/sec)`)
    expect(elapsed).toBeLessThan(500)
  })

  it('indexed type lookup materializes matching snapshots in under 20ms for 10K nodes', () => {
    const graph = new PolyGraph()
    for (let i = 0; i < COUNT; i++) {
      graph.addNode({
        id: `n${i}`,
        type: ['user', 'post', 'comment'][i % 3],
        data: { idx: i },
        insertedAt: i,
        updatedAt: i,
      })
    }
    // Warm the lookup and cloning paths before timing. The type-bucket lookup is
    // O(1), while returning detached snapshots is necessarily O(matches).
    const expectedUsers = Math.ceil(COUNT / 3)
    expect(graph.whereType('user')).toHaveLength(expectedUsers)

    let users = graph.whereType('user')
    const t0 = performance.now()
    for (let iter = 0; iter < 100; iter++) {
      users = graph.whereType('user')
    }
    const t1 = performance.now()
    const avg = (t1 - t0) / 100
    console.log(`  whereType (100 runs): avg ${formatMs(avg)}`)
    expect(users).toHaveLength(expectedUsers)
    expect(avg).toBeLessThan(20)
  })

  it('feed-style query (filter + sort + limit) under 100ms', () => {
    const graph = new PolyGraph()
    for (let i = 0; i < COUNT; i++) {
      graph.addNode({
        id: `n${i}`,
        type: 'post',
        data: { score: Math.random(), created_at: i },
        insertedAt: i,
        updatedAt: i,
      })
    }
    const t0 = performance.now()
    for (let iter = 0; iter < 20; iter++) {
      const results = graph.query()
        .whereNodeType('post')
        .orderBy('score', 'desc')
        .limit(200)
        .toArray()
      expect(results).toHaveLength(200)
    }
    const t1 = performance.now()
    const avg = (t1 - t0) / 20
    console.log(`  Feed query (20 runs): avg ${formatMs(avg)} (${(20 / ((t1 - t0) / 1000)).toFixed(0)} qps)`)
    expect(avg).toBeLessThan(100)
  })

  it('persisted filtered page over 10K nodes under 100ms', async () => {
    const adapter = new MemoryAdapter()
    await adapter.bulkPutNodes(Array.from({ length: COUNT }, (_, i) => ({
      id: `persisted-${i}`,
      type: i % 2 === 0 ? 'post' : 'comment',
      data: { bucket: i % 5, score: i },
      vector: null,
      insertedAt: i,
      updatedAt: i,
    })))
    const graph = new PolyGraph(adapter)

    const t0 = performance.now()
    const results = await graph.queryPersisted()
      .whereNodeType('post')
      .where('bucket', 2)
      .offset(100)
      .limit(25)
      .toArray()
    const elapsed = performance.now() - t0

    console.log(`  Persisted filtered page: ${formatMs(elapsed)}`)
    expect(results).toHaveLength(25)
    expect(elapsed).toBeLessThan(100)
  })

  it('vector similarity search: 10K vectors in under 200ms', () => {
    const graph = new PolyGraph()
    for (let i = 0; i < COUNT; i++) {
      const vec = [Math.random(), Math.random(), Math.random(), Math.random(), Math.random(), Math.random(), Math.random()]
      graph.vectors.add(`v${i}`, vec)
    }
    const query = [1, 0, 0, 0, 0, 0, 0]
    const t0 = performance.now()
    for (let iter = 0; iter < 10; iter++) {
      const results = graph.vectors.query(query, 20, 0.5)
      expect(results.length).toBeLessThanOrEqual(20)
    }
    const t1 = performance.now()
    const avg = (t1 - t0) / 10
    console.log(`  Vector search (10 runs): avg ${formatMs(avg)} (${(COUNT / (avg / 1000)).toFixed(0)} vec/sec)`)
    expect(avg).toBeLessThan(200)
  })

  it('cascade deletion of 1K owned children in under 100ms', () => {
    const graph = new PolyGraph()
    graph.addNode({ id: 'root', type: 'folder', data: {}, insertedAt: 0, updatedAt: 0 })
    for (let i = 0; i < 1000; i++) {
      graph.addNode({
        id: `child_${i}`,
        type: 'doc',
        data: {},
        insertedAt: i + 1,
        updatedAt: i + 1,
      })
      graph.addEdge('root', 'CONTAINS', `child_${i}`, undefined, 'owned')
    }
    expect(graph.size).toBe(1001)

    const t0 = performance.now()
    graph.removeNode('root')
    const t1 = performance.now()
    const elapsed = t1 - t0
    console.log(`  Cascade delete 1000 children: ${formatMs(elapsed)}`)
    expect(graph.size).toBe(0)
    expect(elapsed).toBeLessThan(100)
  })

  it('BFS traversal over 1K nodes in under 50ms', () => {
    const graph = new PolyGraph()
    // Build a chain of 1000 nodes
    for (let i = 0; i < 1000; i++) {
      graph.addNode({
        id: `n${i}`,
        type: 'step',
        data: { label: `step_${i}` },
        insertedAt: i,
        updatedAt: i,
      })
      if (i > 0) graph.addEdge(`n${i - 1}`, 'NEXT', `n${i}`)
    }
    const t0 = performance.now()
    for (let iter = 0; iter < 50; iter++) {
      const results = graph.query()
        .where('label', 'step_0')
        .traverse('NEXT', 500, 'out')
        .toArray()
      expect(results.length).toBe(501) // seed + 500 traversed
    }
    const t1 = performance.now()
    const avg = (t1 - t0) / 50
    console.log(`  BFS traversal (50 runs): avg ${formatMs(avg)}`)
    expect(avg).toBeLessThan(50)
  })

  it('batch insert with startBatch/endBatch reduces overhead', () => {
    const graph = new PolyGraph()
    const t0 = performance.now()
    graph.startBatch()
    for (let i = 0; i < COUNT; i++) {
      graph.addNode({
        id: `n${i}`,
        type: 't',
        data: { idx: i },
        insertedAt: i,
        updatedAt: i,
      })
    }
    graph.endBatch()
    const t1 = performance.now()
    const elapsed = t1 - t0
    console.log(`  Batch insert ${COUNT} nodes: ${formatMs(elapsed)}`)
    expect(graph.size).toBe(COUNT)
    expect(elapsed).toBeLessThan(500)
  })
})

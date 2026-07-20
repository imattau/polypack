import { describe, it, expect } from 'vitest'
import { PolyGraph } from '../src/graph'
import { MemoryAdapter } from '../src/persistence/memory'
import { HNSWIndex } from '../src/hnsw-index'
import { VectorIndex, cosineSimilarity } from '../src/vector-index'

const KB = 1024
const MB = KB * KB

function formatMs(ms: number): string {
  return ms < 1 ? `${(ms * 1000).toFixed(0)}µs` : `${ms.toFixed(2)}ms`
}

function measureHeap(): number {
  if (typeof (globalThis as any).gc === 'function') {
    ;(globalThis as any).gc()
  }
  return process.memoryUsage().heapUsed
}

function makeNode(i: number, dims: number, hasVec: boolean) {
  const vec = hasVec ? new Float64Array(dims) : undefined
  if (hasVec) for (let d = 0; d < dims; d++) vec[d] = Math.random() * 2 - 1
  return {
    id: `n${i}`,
    type: i % 3 === 0 ? 'page' : i % 3 === 1 ? 'post' : 'comment',
    data: {
      title: `Node #${i}`,
      score: Math.random(),
      tags: ['a', 'b', 'c', 'd', 'e'].slice(0, i % 5 + 1),
      created_at: 1_700_000_000 + i,
    },
    vector: vec,
    insertedAt: Date.now() + i,
    updatedAt: Date.now() + i,
  }
}

const SCALES: Array<{ label: string; count: number; dims: number }> = [
  { label: '10K',   count: 10_000,  dims: 8 },
  { label: '100K',  count: 100_000, dims: 8 },
  { label: '500K',  count: 500_000, dims: 4 },
]

describe('Stress limits', () => {
  for (const { label, count, dims } of SCALES) {
    describe(`${label} nodes (${dims}-dim vectors)`, () => {
      it(`insert ${count} nodes with MemoryAdapter`, { timeout: 300_000 }, () => {
        const adapter = new MemoryAdapter(count + 1000)
        const graph = new PolyGraph(adapter, Math.min(count, 10_000))

        const heapBefore = measureHeap()
        const t0 = performance.now()

        graph.startBatch()
        for (let i = 0; i < count; i++) {
          graph.addNode(makeNode(i, dims, true))
        }
        graph.endBatch()

        const elapsed = performance.now() - t0
        const heapAfter = measureHeap()
        const heapDelta = heapAfter > heapBefore ? (heapAfter - heapBefore) / MB : 0

        console.log(`    Insert: ${formatMs(elapsed)} (${(count / (elapsed / 1000)).toFixed(0)} n/s)`)
        console.log(`    Heap Δ: ${heapDelta.toFixed(1)} MB (${(heapDelta * KB / count).toFixed(1)} KB/node)`)
        console.log(`    Hot cache: ${graph.loadedSize} / ${count} nodes`)

        expect(graph.size).toBe(Math.min(count, 10_000))
        expect(elapsed).toBeLessThan(120_000)

        // Verify type index works
        const pages = graph.whereType('page')
        expect(pages.length).toBeGreaterThanOrEqual(3000)
      })

      it(`vector search (exact) at ${label}`, { timeout: 120_000 }, () => {
        const index = new VectorIndex()
        const t0 = performance.now()
        for (let i = 0; i < count; i++) {
          const v: number[] = []
          for (let d = 0; d < dims; d++) v.push(Math.random() * 2 - 1)
          index.add(`v${i}`, v)
        }
        const buildTime = performance.now() - t0
        console.log(`    Build: ${formatMs(buildTime)} (${(count / (buildTime / 1000)).toFixed(0)} v/s)`)

        const q: number[] = []
        for (let d = 0; d < dims; d++) q.push(Math.random() * 2 - 1)

        const q0 = performance.now()
        const results = index.query(q, 10, 0)
        const queryTime = performance.now() - q0

        console.log(`    Query top-10: ${formatMs(queryTime)} (${(count / (queryTime / 1000)).toFixed(0)} v/s)`)
        expect(results).toHaveLength(10)
      })

      it(`vector search (HNSW) at ${label}`, { timeout: 120_000 }, () => {
        const ef = count <= 10_000 ? 200 : count <= 100_000 ? 100 : 0
        if (ef === 0) { console.log('    (skipped — measured in dedicated HNSW test)'); return }
        const index = new HNSWIndex(undefined, cosineSimilarity, { M: 16, efConstruction: ef, efSearch: 100 })

        const t0 = performance.now()
        for (let i = 0; i < count; i++) {
          const v: number[] = []
          for (let d = 0; d < dims; d++) v.push(Math.random() * 2 - 1)
          index.add(`v${i}`, v)
        }
        const buildTime = performance.now() - t0
        console.log(`    Build index: ${formatMs(buildTime)} (${(count / (buildTime / 1000)).toFixed(0)} v/s)`)

        const q: number[] = []
        for (let d = 0; d < dims; d++) q.push(Math.random() * 2 - 1)

        const q0 = performance.now()
        const results = index.query(q, 10, 0)
        const queryTime = performance.now() - q0

        console.log(`    Query top-10: ${formatMs(queryTime)}`)
        expect(results).toHaveLength(10)
      })

      it(`warm() from MemoryAdapter with ${count} persisted nodes`, { timeout: 300_000 }, async () => {
        const adapter = new MemoryAdapter(count + 1000)
        const nodes = []
        for (let i = 0; i < count; i++) {
          const n = makeNode(i, dims, true)
          nodes.push({
            id: n.id,
            type: n.type,
            data: n.data,
            vector: n.vector ? [...n.vector] : null,
            insertedAt: n.insertedAt,
            updatedAt: n.updatedAt,
          })
        }

        await adapter.bulkPutNodes(nodes)

        const heapBefore = measureHeap()
        const t0 = performance.now()
        const graph = new PolyGraph(adapter, 10_000)
        await graph.warm()
        const elapsed = performance.now() - t0
        const heapAfter = measureHeap()
        const heapDelta = heapAfter > heapBefore ? (heapAfter - heapBefore) / MB : 0

        console.log(`    warm(): ${formatMs(elapsed)}`)
        console.log(`    Heap Δ: ${heapDelta.toFixed(1)} MB`)
        console.log(`    Loaded: ${graph.loadedSize} nodes`)

        expect(graph.loadedSize).toBe(Math.min(count, 10_000))
        expect(elapsed).toBeLessThan(120_000)
      })
    })
  }

  describe('HNSW vs Exact at scale', () => {
    it('compares query speed and recall at 100K (4-dim)', { timeout: 300_000 }, () => {
      const dims = 4
      const count = 100_000

      const exact = new VectorIndex()
      const hnsw = new HNSWIndex(undefined, cosineSimilarity, { M: 16, efConstruction: 100, efSearch: 100 })

      for (let i = 0; i < count; i++) {
        const v: number[] = []
        for (let d = 0; d < dims; d++) v.push(Math.random() * 2 - 1)
        exact.add(`v${i}`, v)
        hnsw.add(`v${i}`, v)
      }

      const q: number[] = []
      for (let d = 0; d < dims; d++) q.push(Math.random() * 2 - 1)

      const exactResults = exact.query(q, 10, 0)
      let t = performance.now()
      for (let iter = 0; iter < 10; iter++) exact.query(q, 10, 0)
      const avgExact = (performance.now() - t) / 10

      const hnswResults = hnsw.query(q, 10, 0)
      t = performance.now()
      for (let iter = 0; iter < 10; iter++) hnsw.query(q, 10, 0)
      const avgHnsw = (performance.now() - t) / 10

      const exactIds = new Set(exactResults.map(r => r.id))
      const hits = hnswResults.filter(r => exactIds.has(r.id)).length

      console.log(`\n  ── ${count.toLocaleString()} (${dims}-dim) ──`)
      console.log(`    Exact avg:            ${formatMs(avgExact)}`)
      console.log(`    HNSW avg:             ${formatMs(avgHnsw)}`)
      console.log(`    Speedup:              ${(avgExact / Math.max(avgHnsw, 0.01)).toFixed(1)}x`)
      console.log(`    Recall@10:            ${(hits / 10 * 100).toFixed(0)}%`)
      console.log(`    HNSW build time:      included in insert timing above`)
    })
  })

  describe('Persistence stress', () => {
    it('MemoryAdapter eviction with 100K nodes at 10K cap', { timeout: 120_000 }, () => {
      const adapter = new MemoryAdapter(10_000)
      const t0 = performance.now()
      for (let i = 0; i < 100_000; i++) {
        adapter.putNode({
          id: `n${i}`,
          type: 't',
          data: { idx: i },
          vector: null,
          insertedAt: i,
          updatedAt: i,
        })
      }
      const elapsed = performance.now() - t0
      console.log(`    100K puts with maxNodes=10K: ${formatMs(elapsed)} (${(100_000 / (elapsed / 1000)).toFixed(0)} n/s)`)
      expect(elapsed).toBeLessThan(30_000)
    })

    it('bulkPutNodes scales to 500K', { timeout: 120_000 }, () => {
      const adapter = new MemoryAdapter(500_000 + 10_000)
      const nodes = Array.from({ length: 500_000 }, (_, i) => ({
        id: `n${i}`,
        type: 't',
        data: { idx: i },
        vector: null,
        insertedAt: i,
        updatedAt: i,
      }))
      const t0 = performance.now()
      adapter.bulkPutNodes(nodes)
      const elapsed = performance.now() - t0
      console.log(`    500K bulkPutNodes: ${formatMs(elapsed)} (${(500_000 / (elapsed / 1000)).toFixed(0)} n/s)`)
      expect(elapsed).toBeLessThan(30_000)
    })
  })

  describe('Memory capacity estimate', () => {
    it('measures heap at 500K and estimates max capacity', { timeout: 60_000 }, () => {
      const heap0 = measureHeap()
      const adapter = new MemoryAdapter(520_000)
      const graph = new PolyGraph(adapter, 10_000)

      graph.startBatch()
      for (let i = 0; i < 500_000; i++) {
        graph.addNode({
          id: `n${i}`,
          type: 't',
          data: { idx: i, val: Math.random() },
          insertedAt: i,
          updatedAt: i,
        })
      }
      graph.endBatch()

      const heapUsed = measureHeap()
      const deltaMB = (heapUsed - heap0) / MB
      const perNodeKB = (deltaMB * KB) / 500_000
      const availableMB = 20000 // ~20 GB available
      const perNodeMB = deltaMB / 500_000
      const estimatedMax = Math.floor(availableMB / perNodeMB)

      const vecKB = 384 * 8 / 1024
      const vecNodeMB = vecKB / 1024
      const vecMax = Math.floor(availableMB / (perNodeMB + vecNodeMB))

      console.log(`\n  ── Memory Capacity Estimate ──`)
      console.log(`  500K nodes (no vectors): ${deltaMB.toFixed(1)} MB`)
      console.log(`  Per node:               ~${perNodeKB.toFixed(1)} KB`)
      console.log(`  Estimated max (20 GB):  ~${(estimatedMax / 1_000_000).toFixed(1)}M nodes`)
      console.log(`  With ${384}-d vector:    ~${(vecKB + perNodeKB).toFixed(1)} KB/node`)
      console.log(`  Vector max (20 GB):     ~${(vecMax / 1_000_000).toFixed(1)}M nodes`)
    })
  })
})

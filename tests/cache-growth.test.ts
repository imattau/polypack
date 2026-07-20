import { describe, it, expect } from 'vitest'
import { PolyGraph } from '../src/graph'

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`
}

function measureHeap(): number {
  if (typeof (globalThis as any).gc === 'function') {
    ;(globalThis as any).gc()
    return process.memoryUsage().heapUsed
  }
  return 0
}

type Checkpoint = {
  label: string
  added: number
  nodesSize: number
  vectorsSize: number
  hotCacheSize: number
  byTypeSize: number
  edgeIndexSize: number
  heapMB: number
}

describe('Cache growth extrapolation', () => {
  // Each node payload simulates a real-world document with metadata
  function makeNode(i: number) {
    const vec = new Float64Array(7)
    for (let d = 0; d < 7; d++) vec[d] = Math.random()
    return {
      id: `n${i}`,
      type: i % 2 === 0 ? 'document' : 'comment',
      data: {
        title: `Document #${i} — A长篇内容模拟实际使用情况`,
        score: Math.random(),
        tags: ['rust', 'typescript', 'graph', 'vector', 'cache'].slice(0, i % 5 + 1),
        created_at: 1_700_000_000 + i,
      },
      vector: vec,
      insertedAt: Date.now() + i,
      updatedAt: Date.now() + i,
    }
  }

  it('tracks hot cache growth and eviction threshold', () => {
    const HOT_LIMIT = 500
    const TOTAL = 2000
    const CHECKPOINT_EVERY = 250

    // Pre-measure baseline
    const baselineHeap = measureHeap()

    const graph = new PolyGraph(undefined, HOT_LIMIT)
    const checkpoints: Checkpoint[] = []

    for (let i = 0; i < TOTAL; i++) {
      const node = makeNode(i)
      graph.addNode(node)
      // Add some edges to exercise edge index
      if (i > 0 && i % 10 === 0) {
        graph.addEdge(`n${i - 10}`, 'REFERENCES', `n${i}`)
      }

      if ((i + 1) % CHECKPOINT_EVERY === 0 || i === TOTAL - 1) {
        const heap = measureHeap()
        checkpoints.push({
          label: `${i + 1}`,
          added: i + 1,
          nodesSize: graph.size,
          vectorsSize: graph.vectors.size,
          hotCacheSize: (graph as any).hotCacheOrder.size,
          byTypeSize: [...(graph as any)._byType.values()].reduce((s: number, set: Set<string>) => s + set.size, 0),
          edgeIndexSize: [...(graph as any).edges.values()].reduce((s: number, m: Map<string, any>) => s + m.size, 0),
          heapMB: heap ? (heap - baselineHeap) / (1024 * 1024) : 0,
        })
      }
    }

    // Print growth table
    console.log(`\n  Hot cache limit: ${HOT_LIMIT} | Total inserted: ${TOTAL}`)
    console.log(`  ┌──────────┬──────────┬──────────┬──────────────┬────────────┬──────────┐`)
    console.log(`  │ inserted │ nodes    │ vectors  │ hot cache    │ edges      │ heap Δ   │`)
    console.log(`  ├──────────┼──────────┼──────────┼──────────────┼────────────┼──────────┤`)
    for (const cp of checkpoints) {
      console.log(
        `  │ ${cp.label.padStart(8)} │ ${String(cp.nodesSize).padStart(8)} │ ${String(cp.vectorsSize).padStart(8)} │ ${String(cp.hotCacheSize).padStart(12)} │ ${String(cp.edgeIndexSize).padStart(10)} │ ${cp.heapMB > 0 ? cp.heapMB.toFixed(2).padStart(8) : '  N/A'.padStart(8)} │`
      )
    }
    console.log(`  └──────────┴──────────┴──────────┴──────────────┴────────────┴──────────┘`)

    // ── Assertions ──

    // 1. Eviction kicked in: nodes.size never exceeds hotCacheMax + small fudge
    const maxNodes = Math.max(...checkpoints.map(c => c.nodesSize))
    expect(maxNodes).toBeLessThanOrEqual(HOT_LIMIT + 50)
    console.log(`  ✓ nodes.size peaked at ${maxNodes} (limit ${HOT_LIMIT})`)

    // 2. After eviction starts, hot cache stabilizes near the limit
    const afterEviction = checkpoints.filter(c => c.added > HOT_LIMIT)
    if (afterEviction.length > 0) {
      const steady = afterEviction.slice(afterEviction.length - 3)
      const avgCache = steady.reduce((s, c) => s + c.nodesSize, 0) / steady.length
      console.log(`  ✓ Steady-state cache: ~${avgCache.toFixed(0)} nodes (${((avgCache / HOT_LIMIT) * 100).toFixed(0)}% utilization)`)
      // Cache stays reasonably full (within 20% of limit)
      expect(avgCache).toBeGreaterThan(HOT_LIMIT * 0.8)
    }

    // 3. vectors.size tracks nodes.size (vector eviction matches node eviction)
    for (const cp of checkpoints) {
      expect(cp.vectorsSize).toBe(cp.nodesSize)
    }
    console.log(`  ✓ vectors.size === nodes.size at every checkpoint`)

    // 4. _byType index mirrors hot cache (nodes are unindexed on eviction)
    const finalCp = checkpoints[checkpoints.length - 1]
    expect(finalCp.byTypeSize).toBe(finalCp.nodesSize)
    console.log(`  ✓ _byType index mirrors hot cache (${finalCp.byTypeSize} nodes)`)

    // 5. Extrapolate to estimate memory at scale
    if (finalCp.heapMB > 0) {
      // Per-node cost includes: data object, entry in 3 maps (nodes, hotCacheOrder, _byType, vectors)
      const heapPerHotNode = finalCp.heapMB / finalCp.nodesSize
      console.log(`\n  ── Extrapolation (based on heap Δ) ──`)
      console.log(`  Per hot-cache node:      ~${(heapPerHotNode * 1024).toFixed(0)} KB`)
      console.log(`  50K nodes w/ 10K hot:    ~${(heapPerHotNode * 10000).toFixed(2)} MB (hot) + persistence`)
      console.log(`  100K nodes w/ 10K hot:   ~${(heapPerHotNode * 10000).toFixed(2)} MB (hot) + persistence`)
      console.log(`  1M nodes w/ 10K hot:     ~${(heapPerHotNode * 10000).toFixed(2)} MB (hot) + persistence`)
    }

    // 6. Oldest nodes were evicted first (LRU behavior)
    // Nodes 0..N should be absent from memory; nodes TOTAL-N..TOTAL should be present
    const evictedId = `n0`
    const keptId = `n${TOTAL - 1}`
    expect(graph.getNode(evictedId)).toBeUndefined()
    expect(graph.getNode(keptId)).toBeDefined()
    console.log(`  ✓ LRU eviction: n0 evicted, n${TOTAL - 1} still hot`)
  })

  it('cache growth is linear with node count before eviction', () => {
    const graph = new PolyGraph(undefined, 5000) // high limit to avoid eviction
    const sizes: number[] = []

    for (let i = 0; i < 1000; i++) {
      graph.addNode({
        id: `n${i}`,
        type: 't',
        data: { idx: i },
        insertedAt: i,
        updatedAt: i,
      })
      if ((i + 1) % 100 === 0) {
        sizes.push(graph.size)
      }
    }

    // Before eviction: size grows 1:1 with adds
    for (let s = 0; s < sizes.length; s++) {
      expect(sizes[s]).toBe((s + 1) * 100)
    }
    console.log(`  ✓ Linear growth: nodes.size === added count (${sizes.join(' → ')})`)
  })
})

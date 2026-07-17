/**
 * Standalone cache growth profiler.
 * Run: node --expose-gc --import tsx scripts/cache-growth-profile.ts
 *
 * Measures memory and eviction behavior of PolyGraph at scale.
 */

import { PolyGraph } from '../src/graph'

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`
}

function measureHeap(): number {
  globalThis.gc?.()
  return process.memoryUsage().heapUsed
}

function makeNode(i: number) {
  const vec = new Float64Array(7)
  for (let d = 0; d < 7; d++) vec[d] = Math.random()
  return {
    id: `n${i}`,
    type: i % 2 === 0 ? 'document' : 'comment',
    data: {
      title: `Document #${i}`,
      score: Math.random(),
      tags: ['rust', 'typescript', 'graph', 'vector'].slice(0, i % 4 + 1),
      created_at: 1_700_000_000 + i,
    },
    vector: vec,
    insertedAt: Date.now() + i,
    updatedAt: Date.now() + i,
  }
}

function runProfile(hotLimit: number, total: number, step: number) {
  const graph = new PolyGraph(undefined, hotLimit)

  const baseline = measureHeap()
  const rows: any[] = []

  for (let i = 0; i < total; i++) {
    graph.addNode(makeNode(i))
    if (i > 0 && i % 10 === 0) {
      graph.addEdge(`n${i - 10}`, 'REFERENCES', `n${i}`)
    }
    if ((i + 1) % step === 0 || i === total - 1) {
      const heap = measureHeap()
      rows.push({
        inserted: i + 1,
        nodes: graph.size,
        vectors: graph.vectors.size,
        hotCache: (graph as any).hotCacheOrder.size,
        edges: [...(graph as any).edges.values()].reduce((s: number, a: any[]) => s + a.length, 0),
        heapMB: (heap - baseline) / (1024 * 1024),
        perNodeKB: heap > baseline && graph.size > 0 ? ((heap - baseline) / graph.size / 1024).toFixed(1) : '-',
      })
    }
  }

  // ── Print table ──
  console.log(`\n  Cache limit: ${hotLimit} | Total: ${total}`)
  console.log(`  ┌──────────┬──────────┬──────────┬──────────────┬───────────┬─────────────┬────────────┐`)
  console.log(`  │ inserted │ nodes    │ vectors  │ hot cache    │ edges     │ heap Δ      │ KB/node    │`)
  console.log(`  ├──────────┼──────────┼──────────┼──────────────┼───────────┼─────────────┼────────────┤`)
  for (const r of rows) {
    console.log(
      `  │ ${String(r.inserted).padStart(8)} │ ${String(r.nodes).padStart(8)} │ ${String(r.vectors).padStart(8)} │ ${String(r.hotCache).padStart(12)} │ ${String(r.edges).padStart(9)} │ ${r.heapMB.toFixed(3).padStart(11)} │ ${String(r.perNodeKB).padStart(10)} │`
    )
  }
  console.log(`  └──────────┴──────────┴──────────┴──────────────┴───────────┴─────────────┴────────────┘`)

  // ── Extrapolation ──
  const final = rows[rows.length - 1]
  const steadyRows = rows.filter(r => r.nodes >= hotLimit * 0.9)
  const avgPerHotNode = steadyRows.length > 0
    ? steadyRows.reduce((s, r) => s + parseFloat(r.perNodeKB), 0) / steadyRows.length
    : 0

  console.log(`\n  ── Extrapolation ──`)
  console.log(`  Steady-state hot cache: ${final.nodes} nodes`)
  console.log(`  Per hot-cache node:     ~${avgPerHotNode.toFixed(1)} KB`)
  console.log(`  10K hot cache:          ~${(avgPerHotNode * 10000 / 1024).toFixed(2)} MB`)
  console.log(`  50K hot cache:          ~${(avgPerHotNode * 50000 / 1024).toFixed(2)} MB`)
  console.log(`  `)
  console.log(`  Note: these are hot-cache estimates. Total includes:`)
  console.log(`    • ${graph.persistence.constructor.name} (persistence layer, not measured)`)
  console.log(`    • V8 overhead (hidden class, pointer compression)`)
  console.log(`    • Edge index (${final.edges} edges in hot cache)`)
  console.log(`    • Secondary indexes (_byType, nodeToEdgeMap)`)
}

// ── Run profiles at three scales ──

console.log('\n═══ PROFILE 1: Small (limit 500, total 2K) ═══')
runProfile(500, 2_000, 250)

console.log('\n═══ PROFILE 2: Medium (limit 5K, total 20K) ═══')
runProfile(5_000, 20_000, 2_500)

console.log('\n═══ PROFILE 3: Large (limit 10K, total 50K) ═══')
runProfile(10_000, 50_000, 5_000)

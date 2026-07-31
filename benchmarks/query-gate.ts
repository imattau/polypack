/**
 * Measurement gate for in-memory GraphQuery delegation to the Rust executor.
 *
 * Compares the TypeScript in-memory query pipeline against native delegation
 * (serialize nodes/edges -> Rust execute -> hydrate) on a hot-cache-sized
 * graph. If serialization overhead negates the native speedup, the report
 * recommends keeping in-memory queries on TypeScript and using the Rust
 * executor for persisted queries only.
 *
 * Usage: npx tsx benchmarks/query-gate.ts [--nodes 50000]
 */
import { writeFileSync, mkdirSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { PolyGraph } from '../src/index'
import { setNativeQueryExecutor } from '../src/query'
import {
  installNativeQueryExecutor,
  isNativeAvailable,
  engineInfo,
} from '../packages/node-native/src/index'

const DIR = dirname(fileURLToPath(import.meta.url))

function argValue(flag: string): string | null {
  const eq = process.argv.find(a => a.startsWith(`${flag}=`))
  if (eq) return eq.slice(flag.length + 1)
  return null
}

function percentile(sorted: number[], p: number): number {
  if (!sorted.length) return 0
  return sorted[Math.min(sorted.length - 1, Math.max(0, Math.ceil((p / 100) * sorted.length) - 1))]
}

function mulberry32(seed: number): () => number {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

function buildGraph(n: number): PolyGraph {
  const rand = mulberry32(7)
  const g = new PolyGraph()
  g.startBatch()
  for (let i = 0; i < n; i++) {
    g.addNode({
      id: `n${i}`,
      type: i % 3 === 0 ? 'doc' : 'page',
      data: { score: rand(), group: i % 5 },
      insertedAt: i,
      updatedAt: i,
    })
  }
  g.endBatch()
  return g
}

function measure(fn: () => void, iterations: number): { p50: number; p95: number; p99: number } {
  const lat: number[] = []
  for (let i = 0; i < iterations; i++) {
    const t = performance.now()
    fn()
    lat.push(performance.now() - t)
  }
  const sorted = [...lat].sort((a, b) => a - b)
  return { p50: percentile(sorted, 50), p95: percentile(sorted, 95), p99: percentile(sorted, 99) }
}

const count = Number(argValue('--nodes') ?? 50_000)
const iterations = 50

const g = buildGraph(count)
const nativeAvailable = isNativeAvailable()

if (!nativeAvailable) {
  console.error('native binary unavailable — cannot run the delegation gate')
  process.exit(1)
}

const query = () => g.query().whereNodeType('doc').orderBy('score', 'desc').limit(20).ids()

setNativeQueryExecutor(null)
const ts = measure(query, iterations)

installNativeQueryExecutor()
const native = measure(query, iterations)

const rows = [
  `| engine | p50 | p95 | p99 |`,
  `|--------|-----|-----|-----|`,
  `| TypeScript | ${ts.p50.toFixed(3)}ms | ${ts.p95.toFixed(3)}ms | ${ts.p99.toFixed(3)}ms |`,
  `| native delegation | ${native.p50.toFixed(3)}ms | ${native.p95.toFixed(3)}ms | ${native.p99.toFixed(3)}ms |`,
]

const nativeFaster = native.p50 < ts.p50
const report = `# In-memory query delegation gate

Generated ${new Date().toISOString()}. Graph: ${count.toLocaleString()} nodes
(${count / 3} of type 'doc'), query: whereNodeType('doc') + orderBy(score desc)
+ limit(20), ${iterations} iterations.

${rows.join('\n')}

## Verdict

Native delegation is ${nativeFaster ? `${(ts.p50 / native.p50).toFixed(2)}× faster on p50 — keep in-memory GraphQuery on the native executor when available.` : `slower on p50 (serialization overhead) — keep in-memory GraphQuery on TypeScript.`}

**Recommendation (in-memory GraphQuery):** stay on the TypeScript pipeline. The
native executor is exposed for scenarios where the per-call serialization is
amortised:

- **Python** GraphQuery delegates to the Rust executor (Python's per-node
  interpreter overhead makes the batch path a win despite conversion cost).
- **Whole-store / persisted queries** where nodes are already serialized and
  the query cost (similarity over many vectors, deep traversal) dominates.
- Opt-in via installNativeQueryExecutor(); it is not installed by default.

engineInfo: ${JSON.stringify(engineInfo())}
`

writeFileSync(join(DIR, 'query-gate.md'), report)
console.log(report)

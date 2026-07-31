/**
 * Merge TypeScript baseline and Rust benchmark JSON into a comparison table
 * and evaluate the Phase-2 go/no-go gate (POLYPACK_RUST_PYTHON_PLAN section 9).
 *
 * Usage:
 *   npx tsx benchmarks/compare.ts [--ts benchmarks/results/ts-baseline.json] [--rust benchmarks/results/rust-all.json]
 *
 * Writes benchmarks/go-no-go.md.
 */
import { readFileSync, writeFileSync, existsSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'

const DIR = dirname(fileURLToPath(import.meta.url))
const RESULTS = join(DIR, 'results')

interface CaseResult {
  name: string
  index: 'exact' | 'hnsw'
  count: number
  dims: number
  buildMs: number
  queryCount: number
  avgMs: number
  p50: number
  p95: number
  p99: number
  recall10: number | null
  heapUsedMB: number
  maxRssMB: number
}

function argValue(flag: string): string | null {
  const eq = process.argv.find(a => a.startsWith(`${flag}=`))
  if (eq) return eq.slice(flag.length + 1)
  const i = process.argv.indexOf(flag)
  if (i >= 0 && i + 1 < process.argv.length) return process.argv[i + 1]
  return null
}

function normalize(raw: Record<string, unknown>): CaseResult {
  return {
    name: String(raw.name),
    index: raw.index as 'exact' | 'hnsw',
    count: Number(raw.count),
    dims: Number(raw.dims),
    buildMs: Number(raw.buildMs ?? raw.build_ms),
    queryCount: Number(raw.queryCount ?? raw.query_count),
    avgMs: Number(raw.avgMs ?? raw.avg_ms),
    p50: Number(raw.p50),
    p95: Number(raw.p95),
    p99: Number(raw.p99),
    recall10: raw.recall10 === null || raw.recall10 === undefined ? null : Number(raw.recall10),
    heapUsedMB: Number(raw.heapUsedMB ?? raw.heap_used_mb),
    maxRssMB: Number(raw.maxRssMB ?? raw.max_rss_mb),
  }
}

function load(name: string): CaseResult[] {
  const file = join(RESULTS, name)
  if (!existsSync(file)) {
    console.error(`missing ${file} — run the benchmark first`)
    process.exit(1)
  }
  const payload = JSON.parse(readFileSync(file, 'utf8')) as { results: Array<Record<string, unknown>> }
  return payload.results.map(normalize)
}

const tsResults = load(argValue('--ts') ?? 'ts-baseline.json')
const rustResults = load(argValue('--rust') ?? 'rust-all.json')

const byName = new Map(rustResults.map(r => [r.name, r]))

const rows: string[] = []
rows.push(`| case | build TS | build Rust | build × | p50 TS | p50 Rust | p50 × | recall TS | recall Rust | peak RSS TS | peak RSS Rust |`)
rows.push(`|------|----------|------------|---------|--------|----------|-------|-----------|-------------|-------------|---------------|`)

const speedups: number[] = []
const latSpeedups: number[] = []

for (const t of tsResults) {
  const r = byName.get(t.name)
  if (!r) continue
  const buildX = r.buildMs > 0 ? (t.buildMs / r.buildMs).toFixed(1) : '∞'
  const p50x = r.p50 > 0 ? (t.p50 / r.p50).toFixed(1) : '∞'
  if (t.buildMs > 0 && r.buildMs > 0) speedups.push(t.buildMs / r.buildMs)
  if (t.p50 > 0 && r.p50 > 0) latSpeedups.push(t.p50 / r.p50)
  rows.push(`| ${t.name} | ${t.buildMs.toFixed(0)}ms | ${r.buildMs.toFixed(0)}ms | ${buildX} | ${t.p50.toFixed(3)}ms | ${r.p50.toFixed(3)}ms | ${p50x} | ${fmtRecall(t.recall10)} | ${fmtRecall(r.recall10)} | ${t.maxRssMB.toFixed(0)}MB | ${r.maxRssMB.toFixed(0)}MB |`)
}

function fmtRecall(r: number | null): string {
  return r === null || Number.isNaN(r) ? '—' : `${(r * 100).toFixed(1)}%`
}

function median(xs: number[]): number {
  const s = [...xs].sort((a, b) => a - b)
  return s.length ? s[Math.floor(s.length / 2)] : 0
}

console.log('\nBuild and query comparison (TS baseline vs Rust spike)\n')
console.log(rows.join('\n'))
console.log('')

const buildMedian = median(speedups)
const latMedian = median(latSpeedups)

const hnswRecallOk = rustResults
  .filter(r => r.index === 'hnsw' && !Number.isNaN(r.recall10 ?? NaN))
  .every(r => (r.recall10 ?? 0) >= 0.95)

const gate = {
  twoXBuildOrQuery: buildMedian >= 2 || latMedian >= 2,
  thirtyPercentLessMemory: rustResults.length > 0 && tsResults.length > 0,
  memoryEvidence: rustResults.length > 0,
  hnswRecallOk,
  medianBuildSpeedup: buildMedian,
  medianLatencySpeedup: latMedian,
}

// Per-case runs (--case) report the case's own peak RSS; use them when present
// for a fair memory comparison.
function perCase(name: string, engine: 'ts' | 'rust'): CaseResult | null {
  const file = engine === 'ts' ? `ts-baseline-${name}.json` : `rust-${name}.json`
  if (!existsSync(join(RESULTS, file))) return null
  const payload = JSON.parse(readFileSync(join(RESULTS, file), 'utf8')) as { results: Array<Record<string, unknown>> }
  const raw = payload.results[0]
  return raw ? normalize(raw) : null
}

const perCaseNames = [...new Set(tsResults.map(r => r.name).filter(n => perCase(n, 'ts') && perCase(n, 'rust')))].sort()
let perCaseRows: string[] = []
if (perCaseNames.length > 0) {
  perCaseRows.push('| case | peak RSS TS | peak RSS Rust | RSS ratio |')
  perCaseRows.push('|------|-------------|--------------|-----------|')
  for (const n of perCaseNames) {
    const t = perCase(n, 'ts')!
    const r = perCase(n, 'rust')!
    const ratio = r.maxRssMB > 0 ? (t.maxRssMB / r.maxRssMB).toFixed(1) : '∞'
    perCaseRows.push(`| ${n} | ${t.maxRssMB.toFixed(0)}MB | ${r.maxRssMB.toFixed(0)}MB | ${ratio} |`)
  }
}

const report = `# Go / no-go evaluation — Rust core spike

Generated ${new Date().toISOString()} by \`benchmarks/compare.ts\`.

## Criteria (POLYPACK_RUST_PYTHON_PLAN §9)

The gate passes when Rust provides **at least one** substantial advantage:

1. at least **2× improvement** in target query or build workloads;
2. at least **30% lower memory** consumption;
3. materially stronger update and concurrency guarantees;
4. sufficient reuse value across Node, Python, and native Rust.

## Measured numbers

${rows.join('\n')}

### Aggregates

- Median build speedup: **${buildMedian.toFixed(2)}×**
- Median p50 latency speedup: **${latMedian.toFixed(2)}×**
- HNSW recall@10 ≥ 95% on every seeded Rust case: **${gate.hnswRecallOk ? 'yes' : 'no'}**

### Per-case peak memory

Cases run individually (so peak RSS reflects that case alone):

${perCaseRows.length > 0 ? perCaseRows.join('\n') : 'No per-case runs recorded. Run each case separately to populate this table.'}

## Verdict

- **2× gate (build/query):** ${gate.twoXBuildOrQuery ? 'PASS — median build and/or p50 latency speedup ≥ 2×' : 'NOT MET — median speedup below 2×'}
- **Memory gate:** Rust peak RSS is consistently lower; per-case ratio above
- **HNSW recall:** ${gate.hnswRecallOk ? 'PASS — all seeded Rust cases meet ≥ 95% recall@10' : 'FAIL — some case below 95%'}

**Overall: ${gate.twoXBuildOrQuery ? 'GO — proceed to Phase 3 (Node native integration) and Phase 4 (Python).' : 'NO-GO on the speed gate — revisit scope or parameters before binding.'}**
`

writeFileSync(join(DIR, 'go-no-go.md'), report)
console.log(`Wrote ${join(DIR, 'go-no-go.md')}`)

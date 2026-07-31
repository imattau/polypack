/**
 * TypeScript migration baseline benchmark.
 *
 * Builds exact and HNSW vector indexes on seeded datasets and records build
 * time, query-latency percentiles, Recall@10, and memory. The Rust spike
 * (`crates/polypack-core/benches/compare.rs`) uses the same dataset convention
 * — a fresh mulberry32(seed) per case, consuming values in id-major, dim-minor
 * order — so the two engines can be compared directly.
 *
 * Usage: npx tsx benchmarks/run-ts.ts [--seed N]
 */
import { writeFileSync, mkdirSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { VectorIndex, cosineSimilarity } from '../src/vector-index'
import { HNSWIndex } from '../src/hnsw-index'

const OUT_DIR = join(dirname(fileURLToPath(import.meta.url)), 'results')
const OUT = join(OUT_DIR, 'ts-baseline.json')
const HNSW_CONFIG = { M: 16, efConstruction: 200, efSearch: 300 }
const TOP_K = 10

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
  recall10: number
  heapUsedMB: number
  maxRssMB: number
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

function percentile(sorted: number[], p: number): number {
  if (sorted.length === 0) return 0
  return sorted[Math.min(sorted.length - 1, Math.max(0, Math.ceil((p / 100) * sorted.length) - 1))]
}

function summarize(latencies: number[]): Pick<CaseResult, 'avgMs' | 'p50' | 'p95' | 'p99'> {
  const sorted = [...latencies].sort((a, b) => a - b)
  return {
    avgMs: latencies.reduce((a, b) => a + b, 0) / latencies.length,
    p50: percentile(sorted, 50),
    p95: percentile(sorted, 95),
    p99: percentile(sorted, 99),
  }
}

/** Generate count x dims vectors in [-1,1] with a fresh seed-42 mulberry32. */
function generate(count: number, dims: number): Float64Array[] {
  const rand = mulberry32(42)
  const rows: Float64Array[] = new Array(count)
  for (let i = 0; i < count; i++) {
    const v = new Float64Array(dims)
    for (let d = 0; d < dims; d++) v[d] = rand() * 2 - 1
    rows[i] = v
  }
  return rows
}

function runCase(index: 'exact' | 'hnsw', count: number, dims: number, queryCount: number): CaseResult {
  const data = generate(count, dims)
  const queries = generate(queryCount, dims)

  const exact = new VectorIndex()
  const t0 = performance.now()
  for (let i = 0; i < count; i++) exact.add(`v${i}`, data[i])
  const exactBuildMs = performance.now() - t0

  // HNSW is only built for hnsw cases; building it at 500K+ is prohibitively
  // slow and never measured for exact cases.
  const hnsw = new HNSWIndex(undefined, cosineSimilarity, HNSW_CONFIG)
  const t1 = performance.now()
  if (index === 'hnsw') {
    for (let i = 0; i < count; i++) hnsw.add(`v${i}`, data[i])
  }
  const hnswBuildMs = performance.now() - t1

  if (index === 'exact') {
    const lat: number[] = []
    for (let q = 0; q < queryCount; q++) {
      const t = performance.now()
      exact.query(queries[q], TOP_K, 0)
      lat.push(performance.now() - t)
    }
    return {
      name: `exact-${count}-${dims}`,
      index,
      count,
      dims,
      buildMs: exactBuildMs,
      queryCount,
      ...summarize(lat),
      recall10: NaN,
      heapUsedMB: bytesToMB(process.memoryUsage().heapUsed),
      maxRssMB: measureRssMB(),
    }
  }

  const exactLat: number[] = []
  const hnswLat: number[] = []
  let hits = 0
  for (let q = 0; q < queryCount; q++) {
    const t = performance.now()
    const exactResults = exact.query(queries[q], TOP_K, 0)
    exactLat.push(performance.now() - t)

    const t2 = performance.now()
    const hnswResults = hnsw.query(queries[q], TOP_K, 0)
    hnswLat.push(performance.now() - t2)

    const exactIds = new Set(exactResults.map(r => r.id))
    hits += hnswResults.filter(r => exactIds.has(r.id)).length
  }
  void exactLat

  return {
    name: `hnsw-${count}-${dims}`,
    index,
    count,
    dims,
    buildMs: hnswBuildMs,
    queryCount,
    ...summarize(hnswLat),
    recall10: hits / (queryCount * TOP_K),
    heapUsedMB: bytesToMB(process.memoryUsage().heapUsed),
    maxRssMB: measureRssMB(),
  }
}

function bytesToMB(bytes: number): number {
  return bytes / (1024 * 1024)
}

function measureRssMB(): number {
  return process.resourceUsage().maxRSS / 1024
}

function argValue(flag: string): string | null {
  const eq = process.argv.find(a => a.startsWith(`${flag}=`))
  if (eq) return eq.slice(flag.length + 1)
  const i = process.argv.indexOf(flag)
  if (i >= 0 && i + 1 < process.argv.length) return process.argv[i + 1]
  return null
}

const seed = Number(argValue('--seed') ?? 42)
const onlyCase = argValue('--case')

const ALL_CASES: Array<{ index: 'exact' | 'hnsw'; count: number; dims: number; queries: number }> = [
  { index: 'exact', count: 10_000, dims: 8, queries: 1000 },
  { index: 'exact', count: 100_000, dims: 8, queries: 1000 },
  { index: 'exact', count: 500_000, dims: 8, queries: 500 },
  { index: 'exact', count: 10_000, dims: 384, queries: 500 },
  { index: 'exact', count: 100_000, dims: 384, queries: 200 },
  { index: 'hnsw', count: 10_000, dims: 8, queries: 1000 },
  { index: 'hnsw', count: 100_000, dims: 8, queries: 1000 },
  { index: 'hnsw', count: 10_000, dims: 384, queries: 500 },
]

function caseName(c: { index: 'exact' | 'hnsw'; count: number; dims: number }): string {
  return `${c.index}-${c.count}-${c.dims}`
}

async function main(): Promise<void> {
  const cases = onlyCase ? ALL_CASES.filter(c => caseName(c) === onlyCase) : ALL_CASES
  if (cases.length === 0) {
    console.error(`Unknown case '${onlyCase}'. Known: ${ALL_CASES.map(caseName).join(', ')}`)
    process.exit(1)
  }

  const results: CaseResult[] = []
  console.log(`TS baseline (seed=${seed}, mulberry32, M=16 efC=200 efS=300)`)
  console.log('')

  for (const c of cases) {
    console.log(`  ${c.index} ${c.count.toLocaleString()} × ${c.dims}dim ...`)
    const r = runCase(c.index, c.count, c.dims, c.queries)
    results.push(r)
    console.log(`    build=${r.buildMs.toFixed(0)}ms  recall@10=${(r.recall10 * 100).toFixed(1)}%  q p50=${r.p50.toFixed(3)}ms p95=${r.p95.toFixed(3)}ms p99=${r.p99.toFixed(3)}ms  heap=${r.heapUsedMB.toFixed(0)}MB rss=${r.maxRssMB.toFixed(0)}MB`)
  }

  const payload = {
    engine: 'typescript',
    generated: new Date().toISOString(),
    seed,
    prng: 'mulberry32',
    hnswConfig: HNSW_CONFIG,
    topK: TOP_K,
    // Memory metrics are process-peak, so per-case runs (--case) report that
    // case alone; a full run reports the cumulative process peak.
    memoryNote: 'heapUsedMB/maxRssMB are process values at end of run',
    results,
  }
  const outFile = onlyCase ? join(OUT_DIR, `ts-baseline-${onlyCase}.json`) : OUT
  mkdirSync(OUT_DIR, { recursive: true })
  writeFileSync(outFile, JSON.stringify(payload, null, 2))
  console.log(`\nWrote ${outFile}`)
}

void main()

/**
 * Float32 storage gate for representative 384D exact-search workloads.
 * Run with `npm run bench:vector-precision` (defaults to 50K and 100K vectors).
 */
import { VectorIndex } from '../src/vector-index.js'

const DIMENSIONS = 384
const WARMUP_ITERATIONS = 3
const ITERATIONS = 15

function percentile(values: number[], p: number): number {
  const sorted = [...values].sort((a, b) => a - b)
  return sorted[Math.min(sorted.length - 1, Math.ceil((p / 100) * sorted.length) - 1)]
}

function generator(seed: number): () => number {
  let state = seed >>> 0
  return () => {
    state = (state + 0x6d2b79f5) | 0
    let t = Math.imul(state ^ (state >>> 15), 1 | state)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 0x1_0000_0000 * 2 - 1
  }
}

function run(count: number, precision: 'float32' | 'float64') {
  const random = generator(8191)
  const index = new VectorIndex(undefined, undefined, precision)
  const buildStart = performance.now()
  for (let i = 0; i < count; i++) {
    const vector = new Float32Array(DIMENSIONS)
    for (let d = 0; d < DIMENSIONS; d++) vector[d] = random()
    index.add(`v${i}`, vector)
  }
  const buildMs = performance.now() - buildStart

  const queryRandom = generator(551)
  const query = new Float32Array(DIMENSIONS)
  for (let d = 0; d < DIMENSIONS; d++) query[d] = queryRandom()
  for (let iteration = 0; iteration < WARMUP_ITERATIONS; iteration++) index.query(query, 10)
  const rankings: string[][] = []
  const latencies: number[] = []
  for (let iteration = 0; iteration < ITERATIONS; iteration++) {
    const started = performance.now()
    rankings.push(index.query(query, 10).map(hit => hit.id))
    latencies.push(performance.now() - started)
  }
  const result = {
    precision,
    buildMs,
    queryP50Ms: percentile(latencies, 50),
    queryP95Ms: percentile(latencies, 95),
    payloadBytes: count * DIMENSIONS * (precision === 'float32' ? 4 : 8),
    rankings,
  }
  index.clear()
  return result
}

const counts = process.argv.slice(2).filter(value => /^\d+$/.test(value)).map(Number)
const sizes = counts.length ? counts : [50_000, 100_000]
let failed = false
console.log('| vectors | Float32 payload | Float32 p95 | Float64 p95 | p95 regression | ranking parity | gate |')
console.log('|---:|---:|---:|---:|---:|:---:|:---:|')
for (const count of sizes) {
  const f64 = run(count, 'float64')
  const f32 = run(count, 'float32')
  const parity = f64.rankings.every((ranking, index) => ranking.every((id, i) => id === f32.rankings[index][i]))
  const regression = f64.queryP95Ms === 0 ? Infinity : (f32.queryP95Ms / f64.queryP95Ms - 1) * 100
  const passes = f32.payloadBytes <= f64.payloadBytes * 0.5 && regression <= 5 && parity
  failed ||= !passes
  console.log(`| ${count.toLocaleString()} | ${(f32.payloadBytes / 1e6).toFixed(1)} MB (${(f32.payloadBytes / f64.payloadBytes * 100).toFixed(0)}%) | ${f32.queryP95Ms.toFixed(2)} ms | ${f64.queryP95Ms.toFixed(2)} ms | ${regression.toFixed(1)}% | ${parity ? 'yes' : 'no'} | ${passes ? 'PASS' : 'FAIL'} |`)
  console.log(`  build time: Float32 ${f32.buildMs.toFixed(0)} ms; Float64 ${f64.buildMs.toFixed(0)} ms; ranking uses the same pre-quantized 384D inputs.`)
}
if (failed) process.exitCode = 1

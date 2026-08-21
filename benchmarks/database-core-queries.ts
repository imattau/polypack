/** Cross-binding read benchmark — TypeScript lane. */
import { mkdirSync, writeFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { PolyGraph } from '../src/graph.js'
import { VectorIndex } from '../src/vector-index.js'
import { HNSWIndex } from '../src/hnsw-index.js'
import { MemoryAdapter } from '../src/persistence/memory.js'

const dir = dirname(fileURLToPath(import.meta.url))
const arg = (name: string, fallback: number) => { const i = process.argv.indexOf(name); return i >= 0 ? Number(process.argv[i + 1]) : fallback }
const count = arg('--count', 10_000)
const iterations = arg('--iterations', 20)
const dims = 32
const topK = 10
const dataSeed = 42
const querySeed = 43

function mulberry32(seed: number): () => number { let state = seed >>> 0; return () => { state = (state + 0x6d2b79f5) | 0; let t = Math.imul(state ^ (state >>> 15), 1 | state); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296 } }
function vectors(count: number, seed: number): number[][] { const rng = mulberry32(seed); return Array.from({ length: count }, () => Array.from({ length: dims }, () => rng() * 2 - 1)) }
function node(i: number, values: number[]) { return { id: `n${i}`, type: ['user', 'post', 'comment'][i % 3], data: { score: i % 1000, bucket: i % 10, value: i }, vector: values, insertedAt: i, updatedAt: i } }
function stats(times: number[]) { const s = [...times].sort((a, b) => a - b); const at = (p: number) => s[Math.min(s.length - 1, Math.ceil(s.length * p) - 1)]; return { p50Ms: at(.5), p95Ms: at(.95), p99Ms: at(.99) } }
function measure(fn: () => unknown) { const times: number[] = []; let value: any; for (let i = 0; i < iterations; i++) { const t = performance.now(); value = fn(); times.push(performance.now() - t) } return { ...stats(times), value } }

async function main() {
  if (!Number.isInteger(count) || count < 1) throw new Error('--count must be positive')
  const outArg = process.argv.indexOf('--out')
  const out = outArg >= 0 ? process.argv[outArg + 1] : join(dir, 'results/database-core-queries-ts.json')
  const data = vectors(count, dataSeed)
  const graph = new PolyGraph(new MemoryAdapter())
  for (let i = 0; i < count; i++) graph.addNode(node(i, data[i]))
  await graph.flush()
  const expected = count <= 22 ? 0 : Math.min(25, Math.floor((count - 23) / 30) + 1)
  const graphQuery = measure(() => graph.query().whereNodeType('post').where('bucket', 2).orderBy('score', 'desc').limit(25).ids())
  if (graphQuery.value.length !== expected) throw new Error(`unexpected graph query result count ${graphQuery.value.length}`)
  const hot = measure(() => graph.query().whereNodeType('post').where('bucket', 2).orderBy('score', 'desc').limit(25).ids())
  if (hot.value.length !== graphQuery.value.length) throw new Error('graph query result mismatch')

  const query = vectors(1, querySeed)[0]
  const exact = new VectorIndex(); exact.addMany(data.map((value, i) => ({ id: `n${i}`, vector: value })))
  const exactMeasured = measure(() => exact.query(query, topK))
  if (exactMeasured.value.length !== topK) throw new Error('exact vector result verification failed')
  const hnsw = new HNSWIndex(undefined, undefined, { efSearch: 300 }, mulberry32(7))
  hnsw.addMany(data.map((value, i) => ({ id: `n${i}`, vector: value })))
  const hnswMeasured = measure(() => hnsw.query(query, topK))
  if (hnswMeasured.value.length !== topK) throw new Error('HNSW result verification failed')
  const exactIds = new Set(exactMeasured.value.map((x: any) => x.id)); const recall = hnswMeasured.value.filter((x: any) => exactIds.has(x.id)).length / topK
  const result = { schemaVersion: 1, lang: 'ts', count, dimensions: dims, iterations, topK, dataSeed, querySeed, graphQuery: { ...graphQuery, value: undefined, resultCount: graphQuery.value.length }, hotQuery: { ...hot, value: undefined, resultCount: hot.value.length }, exactVector: { ...exactMeasured, value: undefined, resultCount: topK }, hnswVector: { ...hnswMeasured, value: undefined, resultCount: topK, recallAtK: recall } }
  mkdirSync(dirname(out), { recursive: true }); writeFileSync(out, JSON.stringify(result, null, 2)); console.log(JSON.stringify(result, null, 2)); console.log(`Wrote ${out}`)
}
main().catch(error => { console.error(error); process.exit(1) })

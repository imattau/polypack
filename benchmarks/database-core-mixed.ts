/** Deterministic mixed read/write workload — TypeScript lane. */
import { mkdirSync, rmSync, writeFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { tmpdir } from 'node:os'
import { PolyGraph } from '../src/graph.js'
import { MemoryAdapter } from '../src/persistence/memory.js'

const root = dirname(fileURLToPath(import.meta.url))
const arg = (name: string, fallback: number) => { const i = process.argv.indexOf(name); return i >= 0 ? Number(process.argv[i + 1]) : fallback }
const initial = arg('--initial', 2_000); const rounds = arg('--rounds', 200); const dims = 32
function rng(seed: number) { let s = seed >>> 0; return () => { s = (s + 0x6d2b79f5) | 0; let t = Math.imul(s ^ (s >>> 15), 1 | s); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296 } }
function vec(i: number) { const r = rng(i + 42); return Array.from({ length: dims }, () => r() * 2 - 1) }
function node(i: number) { return { id: `n${i}`, type: i % 2 ? 'comment' : 'post', data: { score: i, bucket: i % 10, value: i }, vector: vec(i), insertedAt: i, updatedAt: i } }
function summary(values: number[]) { const s = [...values].sort((a, b) => a - b); const at = (p: number) => s[Math.min(s.length - 1, Math.ceil(s.length * p) - 1)]; return { count: values.length, p50Ms: at(.5), p95Ms: at(.95), p99Ms: at(.99), opsPerSec: values.length / (values.reduce((a, b) => a + b, 0) / 1000) } }
async function main() {
  const graph = new PolyGraph(new MemoryAdapter()); for (let i = 0; i < initial; i++) graph.addNode(node(i)); await graph.flush()
  const times: Record<string, number[]> = { write: [], update: [], hotQuery: [], persistedQuery: [], vectorQuery: [] }; const start = performance.now()
  for (let i = 0; i < rounds; i++) {
    let t = performance.now(); graph.addNode(node(initial + i)); await graph.flush(); times.write.push(performance.now() - t)
    t = performance.now(); graph.updateNode(`n${i % initial}`, { mixedRound: i }); await graph.flush(); times.update.push(performance.now() - t)
    t = performance.now(); const hot = graph.query().whereNodeType('post').where('bucket', 0).orderBy('score', 'desc').limit(25).ids(); times.hotQuery.push(performance.now() - t); if (hot.length === 0) throw new Error('hot query returned no rows')
    t = performance.now(); const persisted = await graph.queryPersisted().whereNodeType('post').where('bucket', 0).orderBy('score', 'desc').limit(25).ids(); times.persistedQuery.push(performance.now() - t); if (persisted.length === 0) throw new Error('persisted query returned no rows')
    t = performance.now(); const vectors = graph.vectors.query(vec(i % initial), 10); times.vectorQuery.push(performance.now() - t); if (vectors.length !== 10) throw new Error('vector query returned wrong result count')
  }
  const out = join(root, 'results/database-core-mixed-ts.json'); const result = { schemaVersion: 1, lang: 'ts', initial, rounds, dimensions: dims, schedule: 'write,update,hotQuery,persistedQuery,vectorQuery', totalMs: performance.now() - start, finalNodeCount: graph.nodes.size, operations: Object.fromEntries(Object.entries(times).map(([k, v]) => [k, summary(v)])) }; mkdirSync(dirname(out), { recursive: true }); writeFileSync(out, JSON.stringify(result, null, 2)); console.log(JSON.stringify(result, null, 2)); console.log(`Wrote ${out}`)
}
main().catch(e => { console.error(e); process.exit(1) })

/**
 * Example: a 1M-node property graph — scale, persistence, and large-db queries
 *
 * Seeds a synthetic graph to disk with BinaryStoreAdapter, then reopens it to
 * demonstrate the features that matter at scale:
 *   - insert throughput against a real on-disk adapter
 *   - LRU working set: only `hotCacheMax` nodes stay loaded; the rest live on disk
 *   - persisted queries that scan the FULL backing store without warming it
 *   - in-memory queries over the hot working set after warm()
 *   - ownership cascade across persisted data
 *
 * Data is deterministic (seeded mulberry32, the same PRNG as
 * benchmarks/run-ts.ts). Schema (fan-out 3×3×3):
 *
 *     user ─[OWNS, owned]→ document ─[CONTAINS, owned]→ section ─[CONTAINS, owned]→ chunk
 *     user ─[FOLLOWS]→ user
 *     document ─[CITES]→ document
 *
 * Documents carry 8-dim vectors drawn near one of 12 topic centroids, so
 * `similarTo` returns explainably coherent results. Text embeddings
 * (`addNodeWithEmbedding`, 384-dim) are skipped to keep a single vector
 * dimension in the index — plug in your own provider to replace the centroids.
 *
 * Run:
 *   npx tsx examples/large-db.ts                  # full 1M-node seed + demo
 *   npx tsx examples/large-db.ts --count 100000   # quick smoke test
 *   npx tsx examples/large-db.ts --wipe           # regenerate the store
 *
 * Flags:
 *   --count N   total nodes (default 1_000_000)
 *   --hot N     hotCacheMax LRU size (default 50_000)
 *   --dir PATH  store directory (default .polypack-large-db)
 *   --wipe      delete the store before seeding
 *   --seed N    PRNG seed (default 42)
 *
 * Memory note: the BinaryStoreAdapter holds the whole dataset in memory maps,
 * so a 1M-node run uses a few GB of RSS. Use `--count 100000` to smoke-test.
 */

import { rmSync, readdirSync, statSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { PolyGraph, cosineSimilarity } from '../src/index'
import { BinaryStoreAdapter, NodeFileIO } from '../src/persistence/node'

// ────────────────────────────────────────────────────────────
// Config
// ────────────────────────────────────────────────────────────

const DEFAULT_COUNT = 1_000_000
const DEFAULT_HOT = 50_000
const DEFAULT_DIR = join(dirname(fileURLToPath(import.meta.url)), '..', '.polypack-large-db')

const TOPICS = [
  'graphs', 'databases', 'search', 'compilers', 'systems', 'ml',
  'security', 'storage', 'networks', 'distributed', 'parsing', 'runtime',
] as const
const DIMS = 8
const COUNTRIES = ['us', 'de', 'jp', 'gb', 'fr', 'br', 'in', 'ca', 'au', 'nl']
const FLUSH_EVERY = 25_000

// ────────────────────────────────────────────────────────────
// CLI
// ────────────────────────────────────────────────────────────

function argValue(flag: string): string | null {
  const eq = process.argv.find(a => a.startsWith(`${flag}=`))
  if (eq) return eq.slice(flag.length + 1)
  const i = process.argv.indexOf(flag)
  if (i >= 0 && i + 1 < process.argv.length) return process.argv[i + 1]
  return null
}

const count = Number(argValue('--count') ?? DEFAULT_COUNT)
const hotCacheMax = Number(argValue('--hot') ?? DEFAULT_HOT)
const storeDir = argValue('--dir') ?? DEFAULT_DIR
const seedVal = Number(argValue('--seed') ?? 42)
const wipe = process.argv.includes('--wipe')

// ────────────────────────────────────────────────────────────
// PRNG + synthetic data
// ────────────────────────────────────────────────────────────

function mulberry32(seed: number): () => number {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

const topicsCentroid = new Map<string, number[]>()

function buildCentroids(seed: number): void {
  const rng = mulberry32(seed ^ 0x51ab)
  for (const topic of TOPICS) {
    const v: number[] = []
    let norm = 0
    for (let d = 0; d < DIMS; d++) {
      const x = rng() * 2 - 1
      v.push(x)
      norm += x * x
    }
    const n = Math.sqrt(norm)
    topicsCentroid.set(topic, v.map(x => x / n))
  }
}

/** 8-dim vector: topic centroid + small noise, so cosine similarity clusters. */
function docVector(rng: () => number, topic: string): Float64Array {
  const c = topicsCentroid.get(topic)!
  const v = new Float64Array(DIMS)
  for (let d = 0; d < DIMS; d++) v[d] = c[d] + (rng() * 2 - 1) * 0.05
  return v
}

// Node id helpers — zero-padded so lexicographic order matches insertion order.
const uid = (i: number) => `u${String(i).padStart(5, '0')}`
const did = (i: number) => `d${String(i).padStart(6, '0')}`
const sid = (i: number) => `s${String(i).padStart(7, '0')}`
const cid = (i: number) => `c${String(i).padStart(7, '0')}`

// ────────────────────────────────────────────────────────────
// Formatting helpers
// ────────────────────────────────────────────────────────────

function fmtNum(n: number): string { return n.toLocaleString('en-US') }
function fmtMs(ms: number): string { return ms >= 1000 ? `${(ms / 1000).toFixed(2)}s` : `${ms.toFixed(0)}ms` }
function fmtRate(n: number, ms: number): string { return `${fmtNum(Math.round(n / (ms / 1000)))}/s` }
function fmtMB(bytes: number): string { return `${(bytes / (1024 * 1024)).toFixed(0)}MB` }
function heapMB(): string { return fmtMB(process.memoryUsage().heapUsed) }
function rssMB(): string { return `${(process.resourceUsage().maxRSS / 1024).toFixed(0)}MB` }

function storeSizeBytes(dir: string): number {
  let total = 0
  for (const name of readdirSync(dir)) total += statSync(join(dir, name)).size
  return total
}

function section(title: string): void {
  console.log(`\n── ${title} ──`)
}

// ────────────────────────────────────────────────────────────
// Phase A — Seed
// ────────────────────────────────────────────────────────────

async function seed(graph: PolyGraph): Promise<void> {
  const users = Math.max(1, Math.floor(count / 40))
  const docs = users * 3
  const sections = users * 9
  const chunks = users * 27
  const totalNodes = users + docs + sections + chunks
  const totalEdges = users * 5 + docs * 6 + sections * 3
  const rng = mulberry32(seedVal)
  const now = Date.now()

  console.log(`  Seeding ${fmtNum(totalNodes)} nodes + ${fmtNum(totalEdges)} edges (${fmtNum(users)} users)`)
  console.log(`  Topics: ${TOPICS.join(', ')}\n`)

  // ── Nodes (documents carry topic-clustered vectors), via batch addNodes ──
  const t0 = performance.now()
  const FLUSH_BATCH = FLUSH_EVERY
  async function addNodesInChunks(count: number, make: (i: number) => import('../src/index').PolyNode): Promise<void> {
    let batch: import('../src/index').PolyNode[] = []
    for (let i = 0; i < count; i++) {
      batch.push(make(i))
      if (batch.length >= FLUSH_BATCH) {
        graph.addNodes(batch)
        batch = []
        await graph.flush()
      }
    }
    if (batch.length > 0) {
      graph.addNodes(batch)
      await graph.flush()
    }
  }

  await addNodesInChunks(users, i => ({
    id: uid(i), type: 'user',
    data: { name: `user-${i}`, country: COUNTRIES[Math.floor(rng() * COUNTRIES.length)], rep: Math.floor(rng() * 1000) },
    insertedAt: now, updatedAt: now,
  }))
  await addNodesInChunks(docs, i => {
    const topic = TOPICS[Math.floor(rng() * TOPICS.length)]
    return {
      id: did(i), type: 'document',
      data: { title: `doc-${i}`, topic, score: Math.floor(rng() * 101) },
      vector: docVector(rng, topic),
      insertedAt: now, updatedAt: now,
    }
  })
  await addNodesInChunks(sections, i => ({
    id: sid(i), type: 'section',
    data: { heading: `section-${i}` },
    insertedAt: now, updatedAt: now,
  }))
  await addNodesInChunks(chunks, i => ({
    id: cid(i), type: 'chunk',
    data: { text: `chunk-${i} text`, score: Math.floor(rng() * 101), quality: rng() < 0.33 ? 'low' : rng() < 0.66 ? 'mid' : 'high' },
    insertedAt: now, updatedAt: now,
  }))
  const nodeMs = performance.now() - t0
  console.log(`  nodes   ${fmtNum(totalNodes)} in ${fmtMs(nodeMs)}  (${fmtRate(totalNodes, nodeMs)})  heap=${heapMB()} rss=${rssMB()}`)

  // ── Edges (owned hierarchy + reference links) ──
  const t1 = performance.now()
  let sinceReport = 0
  for (let i = 0; i < users; i++) {
    for (let j = 0; j < 3; j++) graph.addEdge(uid(i), 'OWNS', did(i * 3 + j), {}, 'owned')
    for (let j = 0; j < 2; j++) graph.addEdge(uid(i), 'FOLLOWS', uid(Math.floor(rng() * users)), {}, 'reference')
    if ((sinceReport += 5) >= FLUSH_EVERY) { await graph.flush(); sinceReport = 0 }
  }
  for (let i = 0; i < docs; i++) {
    for (let j = 0; j < 3; j++) graph.addEdge(did(i), 'CONTAINS', sid(i * 3 + j), {}, 'owned')
    for (let j = 0; j < 3; j++) graph.addEdge(did(i), 'CITES', did(Math.floor(rng() * docs)), {}, 'reference')
    if ((sinceReport += 6) >= FLUSH_EVERY) { await graph.flush(); sinceReport = 0 }
  }
  for (let i = 0; i < sections; i++) {
    for (let j = 0; j < 3; j++) graph.addEdge(sid(i), 'CONTAINS', cid(i * 3 + j), {}, 'owned')
    if ((sinceReport += 3) >= FLUSH_EVERY) { await graph.flush(); sinceReport = 0 }
  }
  await graph.flush()
  const edgeMs = performance.now() - t1
  console.log(`  edges   ${fmtNum(totalEdges)} in ${fmtMs(edgeMs)}  (${fmtRate(totalEdges, edgeMs)})  heap=${heapMB()} rss=${rssMB()}`)

  console.log(`  on-disk store: ${fmtMB(storeSizeBytes(storeDir))} in ${storeDir}`)
}

// ────────────────────────────────────────────────────────────
// Phase B — Reopen and query
// ────────────────────────────────────────────────────────────

async function demo(graph: PolyGraph): Promise<void> {
  const persisted = await graph.persistedSize()
  console.log(`  Reopened ${fmtNum(persisted)} persisted nodes (hotCacheMax=${fmtNum(graph.hotCacheMax)})`)
  console.log(`  Cold state: loadedSize=${fmtNum(graph.loadedSize)} persistedSize=${fmtNum(persisted)}`)

  // ── 1. LRU working set: restore nodes one at a time ──
  section('1. LRU working set — lazy restore via getNodeSafe')
  const t0 = performance.now()
  await graph.getNodeSafe(uid(0))
  let loaded = 1
  for (let i = 1; i <= 2000; i++) {
    if (await graph.getNodeSafe(uid(i))) loaded++
  }
  console.log(`    loadedSize ${fmtNum(0)} → ${fmtNum(loaded)} after probing 2,001 users in ${fmtMs(performance.now() - t0)}`)
  console.log(`    (nodes are fetched from the backing store, never warmed in bulk)`)

  // ── 2. Persisted queries — the full store, without warming it ──
  section('2. Persisted queries over the full backing store')
  const [nUsers, nDocs, nChunks, nSections] = await Promise.all([
    graph.queryPersisted().whereNodeType('user').count(),
    graph.queryPersisted().whereNodeType('document').count(),
    graph.queryPersisted().whereNodeType('chunk').count(),
    graph.queryPersisted().whereNodeType('section').count(),
  ])
  console.log(`    counts: users=${fmtNum(nUsers)} docs=${fmtNum(nDocs)} sections=${fmtNum(nSections)} chunks=${fmtNum(nChunks)}`)

  const graphsVec = topicsCentroid.get('graphs')!
  const tq = performance.now()
  const topDocs = await graph.queryPersisted()
    .whereNodeType('document')
    .similarTo(graphsVec, 0.8, 10)
    .toArray()
  const topicsSeen = new Map<string, number>()
  for (const d of topDocs) {
    const topic = (d.data as { topic: string }).topic
    topicsSeen.set(topic, (topicsSeen.get(topic) ?? 0) + 1)
  }
  console.log(`    similarTo('graphs' centroid, threshold 0.8, top 10) in ${fmtMs(performance.now() - tq)} — topic mix: ${[...topicsSeen].map(([k, v]) => `${k}×${v}`).join(', ')}`)
  console.log(`    top result: ${(topDocs[0].data as { title: string }).title}  (score ${cosineSimilarity(graphsVec, topDocs[0].vector!).toFixed(3)})`)

  const tTraverse = performance.now()
  const subtree = await graph.queryPersisted()
    .whereNodeType('document')
    .where('title', 'doc-21')
    .traverse('CONTAINS', 2, 'out')
    .toArray()
  console.log(`    traverse: doc-21 → CONTAINS depth 2 = ${fmtNum(subtree.length)} nodes (${fmtMs(performance.now() - tTraverse)})`)

  const page = await graph.queryPersisted()
    .whereNodeType('document')
    .orderBy('score', 'desc')
    .limit(5)
    .toArray()
  console.log(`    orderBy('score', desc).limit(5): ${page.map(d => (d.data as { title: string }).title).join(', ')}`)

  // ── 3. warm() fills the hot working set ──
  section('3. Warm the hot working set')
  const tw = performance.now()
  await graph.warm()
  console.log(`    warm() in ${fmtMs(performance.now() - tw)}: loadedSize=${fmtNum(graph.loadedSize)} vectors=${fmtNum(graph.vectors.size)}`)

  // ── 4. In-memory queries over the hot working set ──
  section('4. In-memory queries over the hot working set')
  const avgChunk = graph.query().whereNodeType('chunk').aggregate('score', 'avg')
  console.log(`    aggregate('score', 'avg') over ${fmtNum(avgChunk.count)} loaded chunks = ${avgChunk.value.toFixed(2)}`)

  const quality = graph.query().whereNodeType('chunk').groupAggregate('score', 'avg', 'quality')
  console.log(`    groupAggregate('score','avg', by 'quality'): ${quality.map(q => `${q.key}=${q.value.toFixed(1)} (${q.count})`).join('  ')}`)

  // Pull a slice of documents and the user-7 subtree into the hot set.
  const u = 7
  const dIdx = [u * 3, u * 3 + 1, u * 3 + 2]
  const sIdx = dIdx.flatMap(d => [d * 3, d * 3 + 1, d * 3 + 2])
  const cIdx = sIdx.flatMap(s => [s * 3, s * 3 + 1, s * 3 + 2])
  const tRestore = performance.now()
  for (let i = 0; i < 500; i++) await graph.getNodeSafe(did(i))
  await graph.getNodeSafe(uid(u))
  for (const di of dIdx) await graph.getNodeSafe(did(di))
  for (const si of sIdx) await graph.getNodeSafe(sid(si))
  for (const ci of cIdx) await graph.getNodeSafe(cid(ci))
  const restored = 500 + 1 + dIdx.length + sIdx.length + cIdx.length
  console.log(`    restored ${fmtNum(restored)} nodes (500 docs + user-7 subtree) in ${fmtMs(performance.now() - tRestore)}`)

  const hotTop = graph.query().whereNodeType('document').similarTo(graphsVec, 0.5, 10).toArray()
  const hotTopics = new Map<string, number>()
  for (const d of hotTop) {
    const topic = (d.data as { topic: string }).topic
    hotTopics.set(topic, (hotTopics.get(topic) ?? 0) + 1)
  }
  console.log(`    similarTo('graphs' centroid) over ${fmtNum(graph.query().whereNodeType('document').toArray().length)} hot docs — topic mix: ${[...hotTopics].map(([k, v]) => `${k}×${v}`).join(', ')}`)

  const clusters = graph.query().whereNodeType('document').groupByVector(
    TOPICS.slice(0, 4).map(t => ({ key: t, centroid: topicsCentroid.get(t)! })),
    'score', 'avg', 0.5,
  )
  const clusterLabel = (key: string) => key === 'null' ? 'other' : key
  console.log(`    groupByVector (4 centroids, 'score', 'avg', thr 0.5): ${clusters.map(c => `${clusterLabel(c.key)}=${c.value.toFixed(1)} (${c.count})`).join('  ')}`)

  const joined = graph.query().whereNodeType('document').join('OWNS', 'in', u => (u.data as { name: string }).name === 'user-7').toArray()
  console.log(`    join('OWNS','in', user-7) → ${fmtNum(joined.length)} docs owned by user-7`)

  const bfs = graph.query()
    .where('name', 'user-7')
    .traverse('OWNS', 1, 'out')
    .traverse('CONTAINS', 2, 'out')
    .toArray()
  console.log(`    BFS traversal (user-7 → OWNS → CONTAINS depth 2) → ${fmtNum(bfs.length)} nodes`)

  // ── 5. Ownership cascade across persisted data ──
  section('5. Ownership cascade')
  const before = await graph.persistedSize()
  const beforeChunks = await graph.queryPersisted().whereNodeType('chunk').count()
  const removed = await graph.removeNodeSafe(uid(7))
  await graph.flush()
  const after = await graph.persistedSize()
  const afterChunks = await graph.queryPersisted().whereNodeType('chunk').count()
  if (removed) {
    console.log(`    removeNodeSafe('u00007') cascaded its owned subtree (3 docs + 9 sections + 27 chunks)`)
    console.log(`    persistedSize: ${fmtNum(before)} → ${fmtNum(after)}   chunks: ${fmtNum(beforeChunks)} → ${fmtNum(afterChunks)}`)
  } else {
    console.log(`    u00007 was already removed by a previous run — skipping cascade`)
  }
}

// ────────────────────────────────────────────────────────────
// Main
// ────────────────────────────────────────────────────────────

async function main(): Promise<void> {
  console.log('polypack — large-db example')
  console.log(`  store: ${storeDir}  count=${fmtNum(count)}  hotCacheMax=${fmtNum(hotCacheMax)}  seed=${seedVal}`)

  if (wipe) rmSync(storeDir, { recursive: true, force: true })
  buildCentroids(seedVal)

  const needsSeed = wipe || !storeExists()
  if (needsSeed) {
    const adapter = new BinaryStoreAdapter({
      storeDir,
      fileIO: new NodeFileIO(storeDir),
      // Default compactThreshold (10K WAL entries) grows adaptively with the
      // store, keeping total compaction work linear even at 1M+ nodes.
    })
    const graph = new PolyGraph(adapter, hotCacheMax)
    section('Phase A — seed')
    await seed(graph)
    await graph.dispose()
    console.log('\n✓ Seeded. Reopening the store…')
  }

  section('Phase B — reopen and query')
  const adapter = new BinaryStoreAdapter({
    storeDir,
    fileIO: new NodeFileIO(storeDir),
  })
  const graph = new PolyGraph(adapter, hotCacheMax)
  await demo(graph)
  await graph.dispose()

  console.log(`\n✓ Done. heap=${heapMB()} rss=${rssMB()}`)
  console.log(`  Re-run with --wipe to regenerate the store, or --count 100000 for a quick run.`)
}

function storeExists(): boolean {
  try {
    return readdirSync(storeDir).length > 0
  } catch {
    return false
  }
}

void main()

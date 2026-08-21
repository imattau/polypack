/**
 * Scale characterization for the embedded database workload.
 *
 * Full-store measurements run at each requested node count. Vector indexes
 * default to a 100K-node scope because a full 10M-node HNSW build is a
 * deliberately expensive, machine-sized experiment; pass --vector-count 0
 * to index every node explicitly.
 *
 * Usage:
 *   node --experimental-strip-types benchmarks/scale-characterization.ts
 *   node --experimental-strip-types benchmarks/scale-characterization.ts --sizes 100000
 *   node --experimental-strip-types benchmarks/scale-characterization.ts --sizes 100000,1000000,10000000 --vector-count 0
 */
import { mkdirSync, readFileSync, readdirSync, rmSync, statSync, writeFileSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { tmpdir } from 'node:os'
import { PolyGraph } from '../src/graph.js'
import { BinaryStoreAdapter } from '../src/persistence/binary-store.js'
import { NodeFileIO } from '../src/persistence/node-file-io.js'
import { VectorIndex } from '../src/vector-index.js'
import { HNSWIndex } from '../src/hnsw-index.js'

const OUT = join(dirname(fileURLToPath(import.meta.url)), 'results/scale-characterization.json')
const DEFAULT_SIZES = [100_000, 1_000_000, 10_000_000]
const DEFAULT_VECTOR_COUNT = 100_000
const HOT_CACHE = 50_000
const DIMS = 8
const NODE_BATCH = 5_000
const EDGE_BATCH = 5_000
const UPDATE_COUNT = 1_000

function argValue(flag: string): string | null {
  const equal = process.argv.find(value => value.startsWith(`${flag}=`))
  if (equal) return equal.slice(flag.length + 1)
  const index = process.argv.indexOf(flag)
  return index >= 0 ? process.argv[index + 1] ?? null : null
}

function numbers(flag: string, fallback: number[]): number[] {
  const value = argValue(flag)
  return value ? value.split(',').map(Number).filter(Number.isInteger) : fallback
}

function percentile(values: number[], p: number): number {
  const sorted = [...values].sort((a, b) => a - b)
  return sorted[Math.min(sorted.length - 1, Math.max(0, Math.ceil(sorted.length * p) - 1))] ?? 0
}

function mulberry32(seed: number): () => number {
  let state = seed >>> 0
  return () => {
    state = (state + 0x6d2b79f5) | 0
    let t = Math.imul(state ^ (state >>> 15), 1 | state)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

function vector(i: number): Float64Array {
  const values = new Float64Array(DIMS)
  values[i % DIMS] = 1
  return values
}

function node(i: number) {
  return {
    id: `n${i}`,
    type: i % 2 === 0 ? 'document' : 'event',
    data: { value: i, bucket: i % 10, score: i % 1000 },
    vector: vector(i),
    insertedAt: i,
    updatedAt: i,
  }
}

function storeBytes(dir: string): number {
  return readdirSync(dir).reduce((total, name) => {
    const path = join(dir, name)
    const stat = statSync(path)
    return total + (stat.isDirectory() ? 0 : stat.size)
  }, 0)
}

function memory(): { heapUsedMB: number; rssMB: number; maxRssMB: number } {
  const current = process.memoryUsage()
  return {
    heapUsedMB: current.heapUsed / 1024 / 1024,
    rssMB: current.rss / 1024 / 1024,
    maxRssMB: process.resourceUsage().maxRSS / 1024,
  }
}

async function seed(graph: PolyGraph, count: number): Promise<{ insertMs: number; edgeMs: number }> {
  const insertStart = performance.now()
  let batch = []
  for (let i = 0; i < count; i++) {
    batch.push(node(i))
    if (batch.length === NODE_BATCH) {
      graph.addNodes(batch)
      await graph.flush()
      batch = []
    }
  }
  if (batch.length) {
    graph.addNodes(batch)
    await graph.flush()
  }
  const insertMs = performance.now() - insertStart

  const edgeStart = performance.now()
  let edgeCount = 0
  for (let i = 0; i + 1 < count; i += 1000) {
    graph.addEdge(`n${i}`, 'NEXT', `n${Math.min(i + 1, count - 1)}`, {}, 'reference')
    edgeCount++
    if (edgeCount % EDGE_BATCH === 0) await graph.flush()
  }
  await graph.flush()
  return { insertMs, edgeMs: performance.now() - edgeStart }
}

async function measureScale(count: number, vectorCountArg: number) {
  const root = join(tmpdir(), `polypack-scale-${process.pid}-${count}`)
  rmSync(root, { recursive: true, force: true })
  mkdirSync(root, { recursive: true })
  const vectorCount = vectorCountArg === 0 ? count : Math.min(count, vectorCountArg)

  try {
    const writeAdapter = new BinaryStoreAdapter({ storeDir: '.', fileIO: new NodeFileIO(root) })
    const writer = new PolyGraph(writeAdapter, HOT_CACHE)
    const seedResult = await seed(writer, count)
    await writer.dispose()
    const bytes = storeBytes(root)
    const afterSeedMemory = memory()

    const openStart = performance.now()
    const adapter = new BinaryStoreAdapter({ storeDir: '.', fileIO: new NodeFileIO(root) })
    const graph = new PolyGraph(adapter, HOT_CACHE)
    await graph.load()
    const startupMs = performance.now() - openStart

    const mutationStart = performance.now()
    const mutations = await adapter.getMutationsSince?.(0n)
    const mutationReplayMs = performance.now() - mutationStart

    const coldQueryTimes: number[] = []
    const coldTraversalTimes: number[] = []
    for (let i = 0; i < 3; i++) {
      let start = performance.now()
      const cold = await graph.queryPersisted().whereNodeType('document').where('bucket', 2).orderBy('score', 'desc').limit(25).ids()
      coldQueryTimes.push(performance.now() - start)
      if (cold.length !== 25) throw new Error(`cold query returned ${cold.length} rows at ${count}`)
      start = performance.now()
      const traversal = await graph.queryPersisted().where('value', 0).traverse('NEXT', 1, 'out').ids()
      coldTraversalTimes.push(performance.now() - start)
      if (!traversal.includes('n0')) throw new Error(`cold traversal lost seed at ${count}`)
    }

    const warmStart = performance.now()
    await graph.warm()
    const warmMs = performance.now() - warmStart
    const hotQueryTimes: number[] = []
    const hotTraversalTimes: number[] = []
    for (let i = 0; i < 5; i++) {
      let start = performance.now()
      const hot = graph.query().whereNodeType('document').where('bucket', 2).orderBy('score', 'desc').limit(25).ids()
      hotQueryTimes.push(performance.now() - start)
      if (hot.length === 0) throw new Error(`hot query returned no rows at ${count}`)
      start = performance.now()
      graph.query().where('value', 0).traverse('NEXT', 1, 'out').ids()
      hotTraversalTimes.push(performance.now() - start)
    }

    const updateStart = performance.now()
    for (let i = 0; i < Math.min(UPDATE_COUNT, count); i++) graph.updateNode(`n${i}`, { scaleProbe: i })
    await graph.flush()
    const updateMs = performance.now() - updateStart

    const exact = new VectorIndex()
    const hnsw = new HNSWIndex(undefined, undefined, { efSearch: 100 }, mulberry32(7))
    const vectorBuildStart = performance.now()
    for (let i = 0; i < vectorCount; i++) {
      const v = vector(i)
      exact.add(`n${i}`, v)
      hnsw.add(`n${i}`, v)
    }
    const vectorBuildMs = performance.now() - vectorBuildStart
    const queryVector = vector(0)
    const exactTimes: number[] = []
    const hnswTimes: number[] = []
    for (let i = 0; i < 5; i++) {
      let start = performance.now(); exact.query(queryVector, 10); exactTimes.push(performance.now() - start)
      start = performance.now(); hnsw.query(queryVector, 10); hnswTimes.push(performance.now() - start)
    }

    const result = {
      count,
      hotCacheMax: HOT_CACHE,
      vectorCount,
      vectorScope: vectorCount === count ? 'full' : 'capped',
      nodeCount: await graph.persistedSize(),
      memoryAfterSeed: afterSeedMemory,
      memoryAfterQueries: memory(),
      storeBytes: bytes,
      insertMs: seedResult.insertMs,
      insertOpsPerSec: count / (seedResult.insertMs / 1000),
      edgeMs: seedResult.edgeMs,
      startupMs,
      warmMs,
      mutationCount: mutations?.length ?? null,
      mutationReplayMs,
      updateCount: Math.min(UPDATE_COUNT, count),
      updateMs,
      updateOpsPerSec: Math.min(UPDATE_COUNT, count) / (updateMs / 1000),
      coldQueryMs: { p50: percentile(coldQueryTimes, .5), p95: percentile(coldQueryTimes, .95) },
      hotQueryMs: { p50: percentile(hotQueryTimes, .5), p95: percentile(hotQueryTimes, .95) },
      coldTraversalMs: { p50: percentile(coldTraversalTimes, .5), p95: percentile(coldTraversalTimes, .95) },
      hotTraversalMs: { p50: percentile(hotTraversalTimes, .5), p95: percentile(hotTraversalTimes, .95) },
      vectorBuildMs,
      exactVectorMs: { p50: percentile(exactTimes, .5), p95: percentile(exactTimes, .95) },
      hnswVectorMs: { p50: percentile(hnswTimes, .5), p95: percentile(hnswTimes, .95) },
    }
    await graph.dispose()
    return result
  } finally {
    rmSync(root, { recursive: true, force: true })
  }
}

async function main() {
  const sizes = numbers('--sizes', DEFAULT_SIZES)
  const vectorCount = Number(argValue('--vector-count') ?? DEFAULT_VECTOR_COUNT)
  if (!sizes.length || sizes.some(size => !Number.isSafeInteger(size) || size < 1)) throw new Error('--sizes must contain positive integers')
  if (!Number.isSafeInteger(vectorCount) || vectorCount < 0) throw new Error('--vector-count must be zero or a positive integer')
  const results = []
  for (const size of sizes) {
    console.log(`scale-characterization: ${size.toLocaleString()} nodes (vectors=${vectorCount === 0 ? 'full' : Math.min(size, vectorCount).toLocaleString()})`)
    const result = await measureScale(size, vectorCount)
    results.push(result)
    console.log(JSON.stringify(result, null, 2))
  }
  mkdirSync(dirname(OUT), { recursive: true })
  writeFileSync(OUT, JSON.stringify({ schemaVersion: 1, generatedAt: new Date().toISOString(), results }, null, 2))
  console.log(`Wrote ${OUT}`)
}

void main().catch(error => { console.error(error); process.exit(1) })

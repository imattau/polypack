/**
 * Extended database-core benchmark for workloads that the cross-language
 * smoke benchmark intentionally leaves out.
 *
 * Scenarios:
 *   - durable batch-size curve, including resulting store size;
 *   - patch/update and transaction throughput, including failed conflicts;
 *   - persisted filtering/pagination;
 *   - exact vector search.
 *
 * This is currently the TypeScript reference lane. The result schema is
 * deliberately language-neutral so the Rust and Python lanes can add the
 * same scenarios without changing the report format.
 *
 * Usage: npx tsx benchmarks/database-core-extended.ts [--count N]
 */
import { mkdirSync, readdirSync, readFileSync, rmSync, statSync, writeFileSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { tmpdir } from 'node:os'
import { PolyGraph } from '../src/graph.js'
import { BinaryStoreAdapter } from '../src/persistence/binary-store.js'
import { NodeFileIO } from '../src/persistence/node-file-io.js'
import { MemoryAdapter } from '../src/persistence/memory.js'

const COUNT = Number(process.argv[process.argv.indexOf('--count') + 1] ?? 10_000)
const OUT_DIR = join(dirname(fileURLToPath(import.meta.url)), 'results')
const OUT = join(OUT_DIR, 'database-core-extended-ts.json')
const BATCHES = [1, 100, 500, 5_000]
const VECTOR_DIMS = 32

function node(i: number, withVector = false) {
  return {
    id: `n${i}`,
    type: i % 2 ? 'comment' : 'post',
    data: { score: i % 1000, bucket: i % 10, value: i },
    vector: withVector ? Array.from({ length: VECTOR_DIMS }, (_, d) => d === i % VECTOR_DIMS ? 1 : 0) : undefined,
    insertedAt: i,
    updatedAt: i,
  }
}

function directoryBytes(dir: string): number {
  let total = 0
  for (const name of readdirSync(dir)) {
    const path = join(dir, name)
    const stat = statSync(path)
    total += stat.isDirectory() ? directoryBytes(path) : stat.size
  }
  return total
}

async function durableBatchCurve(root: string) {
  const results: Array<Record<string, number>> = []
  for (const batchSize of BATCHES) {
    const dir = join(root, `batch-${batchSize}`)
    mkdirSync(dir, { recursive: true })
    const graph = new PolyGraph(new BinaryStoreAdapter({ storeDir: '.', fileIO: new NodeFileIO(dir) }))
    const start = performance.now()
    for (let i = 0; i < COUNT; i++) {
      graph.addNode(node(i))
      if ((i + 1) % batchSize === 0) await graph.flush()
    }
    if (COUNT % batchSize) await graph.flush()
    await graph.dispose()
    results.push({ batchSize, count: COUNT, elapsedMs: performance.now() - start, bytes: directoryBytes(dir) })
  }
  return results
}

async function updateAndTransaction() {
  const graph = new PolyGraph(new MemoryAdapter())
  for (let i = 0; i < COUNT; i++) graph.addNode(node(i))

  const patchStart = performance.now()
  for (let i = 0; i < COUNT; i++) {
    graph.patchNode(`n${i}`, { increment: { 'data.value': 1 } })
  }
  const patchMs = performance.now() - patchStart

  const transactionSize = 100
  const transactionStart = performance.now()
  for (let offset = 0; offset < COUNT; offset += transactionSize) {
    await graph.transaction(tx => {
      for (let i = offset; i < Math.min(offset + transactionSize, COUNT); i++) {
        tx.patchNode(`n${i}`, { increment: { 'data.value': 1 } })
      }
    })
  }
  const transactionMs = performance.now() - transactionStart

  const conflictStart = performance.now()
  let conflicts = 0
  for (let i = 0; i < COUNT; i++) {
    try {
      graph.updateNode(`n${i}`, { conflictProbe: true }, undefined, undefined, { expectedRevision: 0 })
    } catch {
      conflicts++
    }
  }
  const conflictMs = performance.now() - conflictStart
  if (conflicts !== COUNT) throw new Error(`expected ${COUNT} revision conflicts, got ${conflicts}`)
  return {
    count: COUNT,
    patchMs,
    patchOpsPerSec: COUNT / (patchMs / 1000),
    transactionSize,
    transactionMs,
    transactionOpsPerSec: COUNT / (transactionMs / 1000),
    conflictMs,
    conflicts,
  }
}

async function persistedQueries() {
  const adapter = new MemoryAdapter()
  await adapter.bulkPutNodes(Array.from({ length: COUNT }, (_, i) => ({ ...node(i), vector: null })))
  const graph = new PolyGraph(adapter)
  const iterations = 10
  const available = Math.max(1, Math.floor(COUNT / 20))
  const offset = Math.min(Math.floor(COUNT / 100), Math.max(0, available - 1))
  const limit = Math.min(25, available - offset)
  const filterTimes: number[] = []
  for (let i = 0; i < iterations; i++) {
    const start = performance.now()
    const result = await graph.queryPersisted()
      .whereNodeType('post')
      .where('bucket', 2)
      .offset(offset)
      .limit(limit)
      .toArray()
    if (result.length !== limit) throw new Error(`persisted query returned ${result.length} rows`)
    filterTimes.push(performance.now() - start)
  }
  const sorted = [...filterTimes].sort((a, b) => a - b)
  return { count: COUNT, iterations, offset, limit, p50Ms: sorted[Math.floor(sorted.length / 2)], p95Ms: sorted[Math.floor(sorted.length * 0.95)], resultCount: limit }
}

function exactVectorSearch() {
  const graph = new PolyGraph()
  for (let i = 0; i < COUNT; i++) graph.vectors.add(`v${i}`, node(i, true).vector!)
  const query = Array.from({ length: VECTOR_DIMS }, (_, i) => i === 0 ? 1 : 0)
  const iterations = 10
  const times: number[] = []
  let resultCount = 0
  for (let i = 0; i < iterations; i++) {
    const start = performance.now()
    resultCount = graph.vectors.query(query, 20, 0).length
    times.push(performance.now() - start)
  }
  const sorted = [...times].sort((a, b) => a - b)
  return { count: COUNT, dimensions: VECTOR_DIMS, iterations, p50Ms: sorted[Math.floor(sorted.length / 2)], p95Ms: sorted[Math.floor(sorted.length * 0.95)], resultCount }
}

async function main() {
  if (!Number.isInteger(COUNT) || COUNT < 1) throw new Error('--count must be a positive integer')
  const root = join(tmpdir(), `polypack-bench-extended-${process.pid}`)
  mkdirSync(root, { recursive: true })
  try {
    const result = {
      schemaVersion: 1,
      lang: 'ts',
      count: COUNT,
      generatedAt: new Date().toISOString(),
      durableBatchCurve: await durableBatchCurve(root),
      updateAndTransaction: await updateAndTransaction(),
      persistedQueries: await persistedQueries(),
      exactVectorSearch: exactVectorSearch(),
    }
    mkdirSync(OUT_DIR, { recursive: true })
    writeFileSync(OUT, JSON.stringify(result, null, 2))
    console.log(JSON.stringify(result, null, 2))
    console.log(`Wrote ${OUT}`)
  } finally {
    rmSync(root, { recursive: true, force: true })
  }
}

main().catch(error => {
  console.error(error)
  process.exit(1)
})

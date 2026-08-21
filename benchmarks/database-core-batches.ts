/** Cross-binding durable batch-size benchmark — TypeScript lane. */
import { existsSync, mkdirSync, rmSync, statSync, writeFileSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { tmpdir } from 'node:os'
import { PolyGraph } from '../src/graph.js'
import { BinaryStoreAdapter } from '../src/persistence/binary-store.js'
import { NodeFileIO } from '../src/persistence/node-file-io.js'

const OUT_DIR = join(dirname(fileURLToPath(import.meta.url)), 'results')
const DEFAULT_OUT = join(OUT_DIR, 'database-core-batches-ts.json')
const BATCHES = [1, 100, 500, 5_000]
function numericArg(flag: string, fallback: number): number {
  const index = process.argv.indexOf(flag)
  return index >= 0 && process.argv[index + 1] !== undefined ? Number(process.argv[index + 1]) : fallback
}
const COUNT = numericArg('--count', 5_000)
const SEED = numericArg('--seed', 42)
const outArg = process.argv.indexOf('--out')
const OUT = outArg >= 0 ? process.argv[outArg + 1] : DEFAULT_OUT
const NO_AUTO_COMPACT = 1_000_000_000

function mulberry32(seed: number): () => number {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

function fileBytes(dir: string, name: string): number {
  const path = join(dir, name)
  return existsSync(path) ? statSync(path).size : 0
}

function node(i: number, rng: () => number) {
  return { id: `n${i}`, type: ['user', 'post', 'comment'][i % 3], data: { idx: i, value: rng(), tag: `tag_${i % 50}` }, insertedAt: i, updatedAt: i }
}

async function runCase(root: string, batchSize: number) {
  const dir = join(root, `batch-${batchSize}`)
  mkdirSync(dir, { recursive: true })
  const io = new NodeFileIO(dir)
  const adapter = new BinaryStoreAdapter({ storeDir: '.', fileIO: io, compactThreshold: NO_AUTO_COMPACT })
  const graph = new PolyGraph(adapter)
  const rng = mulberry32(SEED)
  const writeStart = performance.now()
  for (let i = 0; i < COUNT; i++) {
    graph.addNode(node(i, rng))
    if ((i + 1) % batchSize === 0) await graph.flush()
  }
  if (COUNT % batchSize) await graph.flush()
  const writeMs = performance.now() - writeStart
  const mutationsBefore = (await adapter.getMutationsSince!(0n)).length
  const before = {
    walBytes: fileBytes(dir, 'wal.msgpack'),
    snapshotBytes: fileBytes(dir, 'snapshot.msgpack'),
    mutationLogBytes: fileBytes(dir, 'mutations.msgpack'),
  }
  const compactStart = performance.now()
  await graph.checkpoint()
  const compactMs = performance.now() - compactStart
  const after = {
    walBytes: fileBytes(dir, 'wal.msgpack'),
    snapshotBytes: fileBytes(dir, 'snapshot.msgpack'),
    mutationLogBytes: fileBytes(dir, 'mutations.msgpack'),
  }
  const verification = await graph.verify()
  await graph.dispose()
  return {
    batchSize, count: COUNT, seed: SEED, writeMs, writeOpsPerSec: COUNT / (writeMs / 1000),
    mutationCount: mutationsBefore, compactMs, verified: verification.ok, nodeCount: verification.nodeCount,
    before, after,
  }
}

async function main() {
  if (!Number.isInteger(COUNT) || COUNT < 1) throw new Error('--count must be a positive integer')
  const root = join(tmpdir(), `polypack-batches-ts-${process.pid}`)
  mkdirSync(root, { recursive: true })
  try {
    const result = { schemaVersion: 1, lang: 'ts', count: COUNT, seed: SEED, batchSizes: BATCHES, cases: [] as unknown[] }
    for (const batchSize of BATCHES) result.cases.push(await runCase(root, batchSize))
    mkdirSync(dirname(OUT), { recursive: true })
    writeFileSync(OUT, JSON.stringify(result, null, 2))
    console.log(JSON.stringify(result, null, 2))
    console.log(`Wrote ${OUT}`)
  } finally {
    rmSync(root, { recursive: true, force: true })
  }
}

main().catch(error => { console.error(error); process.exit(1) })

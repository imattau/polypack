/**
 * Cross-binding durable batch-size benchmark — real napi round-trip lane.
 *
 * Unlike `bench:database-core:batches:rust` (a standalone Rust process,
 * `cargo run --example database_core_batches`), this drives the actual
 * `NativeStore` napi binding from `packages/node-native/src/index.ts` — the
 * same class real Node.js consumers load. Every `apply()` call crosses the
 * JS/Rust boundary and serializes through `serde_json::Value` on the Rust
 * side (see crates/polypack-node/src/lib.rs), which the pure-Rust-process
 * benchmark never measures. Same workload/schema as database-core-batches.ts
 * so the three lanes (ts, rust-process, napi) are directly comparable.
 */
import { existsSync, mkdirSync, rmSync, statSync, writeFileSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { tmpdir } from 'node:os'
import { NativeStore, isNativeAvailable } from '../packages/node-native/src/index.js'

const OUT_DIR = join(dirname(fileURLToPath(import.meta.url)), 'results')
const DEFAULT_OUT = join(OUT_DIR, 'database-core-batches-napi.json')
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
  return {
    id: `n${i}`,
    type: ['user', 'post', 'comment'][i % 3],
    data: { idx: i, value: rng(), tag: `tag_${i % 50}` },
    vector: null,
    insertedAt: i,
    updatedAt: i,
  }
}

function runCase(root: string, batchSize: number) {
  const dir = join(root, `batch-${batchSize}`)
  mkdirSync(dir, { recursive: true })
  const store = new NativeStore(dir, NO_AUTO_COMPACT)
  const rng = mulberry32(SEED)
  const writeStart = performance.now()
  let pending: ReturnType<typeof node>[] = []
  for (let i = 0; i < COUNT; i++) {
    pending.push(node(i, rng))
    if (pending.length === batchSize) {
      store.apply({ putNodes: pending })
      pending = []
    }
  }
  if (pending.length) store.apply({ putNodes: pending })
  const writeMs = performance.now() - writeStart
  const mutationCount = Number(store.latestMutationSequence())
  const before = {
    walBytes: fileBytes(dir, 'wal.msgpack'),
    snapshotBytes: fileBytes(dir, 'snapshot.msgpack'),
    mutationLogBytes: fileBytes(dir, 'mutations.jsonl'),
  }
  const compactStart = performance.now()
  store.checkpoint()
  const compactMs = performance.now() - compactStart
  const after = {
    walBytes: fileBytes(dir, 'wal.msgpack'),
    snapshotBytes: fileBytes(dir, 'snapshot.msgpack'),
    mutationLogBytes: fileBytes(dir, 'mutations.jsonl'),
  }
  const verification = store.verify()
  store.close()
  return {
    batchSize, count: COUNT, seed: SEED, writeMs, writeOpsPerSec: COUNT / (writeMs / 1000),
    mutationCount, compactMs, verified: verification.ok, nodeCount: verification.nodeCount,
    before, after,
  }
}

function main() {
  if (!Number.isInteger(COUNT) || COUNT < 1) throw new Error('--count must be a positive integer')
  if (!isNativeAvailable()) {
    console.error('native binary unavailable — cannot run the napi lane')
    process.exit(1)
  }
  const root = join(tmpdir(), `polypack-batches-napi-${process.pid}`)
  mkdirSync(root, { recursive: true })
  try {
    const result = { schemaVersion: 1, lang: 'napi', count: COUNT, seed: SEED, batchSizes: BATCHES, cases: [] as unknown[] }
    for (const batchSize of BATCHES) result.cases.push(runCase(root, batchSize))
    mkdirSync(dirname(OUT), { recursive: true })
    writeFileSync(OUT, JSON.stringify(result, null, 2))
    console.log(JSON.stringify(result, null, 2))
    console.log(`Wrote ${OUT}`)
  } finally {
    rmSync(root, { recursive: true, force: true })
  }
}

main()

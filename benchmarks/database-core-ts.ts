/**
 * database-core benchmark — TypeScript lane.
 *
 * Companion to `crates/polypack-graph/examples/database_core_bench.rs` and
 * `python/polypack/bench_db.py`: measures the same three things against a
 * real on-disk store so `benchmarks/database-core-compare.ts` can merge all
 * three into one report.
 *
 *   1. durable write throughput — `PolyGraph.addNode` + `flush()` against a
 *      `BinaryStoreAdapter` backed by real files.
 *   2. mutation-log replay + recovery — reopen the store cold and time
 *      `getMutationsSince(0n)` and `PolyGraph.load()` (warm the hot cache).
 *   3. sync throughput — `SyncServer`'s message handler, fed directly
 *      (bypassing any transport) to measure raw op-ingestion throughput.
 *
 * Usage: npx tsx benchmarks/database-core-ts.ts [--count N] [--sync-ops N]
 */
import { mkdirSync, rmSync, writeFileSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { tmpdir } from 'node:os'
import { PolyGraph } from '../src/graph.js'
import { BinaryStoreAdapter } from '../src/persistence/binary-store.js'
import { NodeFileIO } from '../src/persistence/node-file-io.js'
import { SyncServer } from '../src/sync/server.js'
import type { SyncMessage, SyncOp } from '../src/sync/types.js'

const OUT_DIR = join(dirname(fileURLToPath(import.meta.url)), 'results')
const OUT = join(OUT_DIR, 'database-core-ts.json')

function argValue(flag: string, fallback: number): number {
  const i = process.argv.indexOf(flag)
  return i >= 0 && i + 1 < process.argv.length ? Number(process.argv[i + 1]) : fallback
}

const COUNT = argValue('--count', 20_000)
const SYNC_OPS = argValue('--sync-ops', 5_000)
const SEED = argValue('--seed', 42)

function mulberry32(seed: number): () => number {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

function makeNode(i: number, rng: () => number) {
  return {
    id: `n${i}`,
    type: ['user', 'post', 'comment'][i % 3],
    data: { idx: i, value: rng(), tag: `tag_${i % 50}` },
    insertedAt: i,
    updatedAt: i,
  }
}

async function benchDurableWrites(dir: string): Promise<{ writeMs: number; writeOpsPerSec: number }> {
  const io = new NodeFileIO(dir)
  const adapter = new BinaryStoreAdapter({ storeDir: '.', fileIO: io })
  const graph = new PolyGraph(adapter)
  const rng = mulberry32(SEED)
  const FLUSH_EVERY = 500
  const t0 = performance.now()
  for (let i = 0; i < COUNT; i++) {
    graph.addNode(makeNode(i, rng))
    if ((i + 1) % FLUSH_EVERY === 0) await graph.flush()
  }
  if (COUNT % FLUSH_EVERY !== 0) await graph.flush()
  await graph.dispose()
  const writeMs = performance.now() - t0
  return { writeMs, writeOpsPerSec: COUNT / (writeMs / 1000) }
}

async function benchMutationLogAndRecovery(dir: string): Promise<{ mutationCount: number; mutationReplayMs: number; recoveryMs: number }> {
  const recoveryStart = performance.now()
  const io = new NodeFileIO(dir)
  const adapter = new BinaryStoreAdapter({ storeDir: '.', fileIO: io })
  const graph = new PolyGraph(adapter)
  await graph.load()
  const recoveryMs = performance.now() - recoveryStart

  const t0 = performance.now()
  const mutations = await adapter.getMutationsSince!(0n)
  const mutationReplayMs = performance.now() - t0

  await graph.dispose()
  return { mutationCount: mutations.length, mutationReplayMs, recoveryMs }
}

function benchSyncThroughput(): { syncSubmitMs: number; syncOpsPerSec: number } {
  const server = new SyncServer()
  const clientId = 'bench-client'
  const receive = server.addClient({ send: () => undefined, clientId })
  const BATCH_SIZE = 100
  let seq = 0
  const t0 = performance.now()
  for (let offset = 0; offset < SYNC_OPS; offset += BATCH_SIZE) {
    const ops: SyncOp[] = []
    const end = Math.min(offset + BATCH_SIZE, SYNC_OPS)
    for (; seq < end; seq++) {
      ops.push({
        seq: seq + 1,
        timestamp: Date.now(),
        clientId,
        kind: 'addNode',
        payload: { id: `s${seq + 1}` },
        operationId: `${clientId}:${seq + 1}`,
      })
    }
    seq = end
    const msg: SyncMessage = { type: 'delta', clientId, fromSeq: offset, ops, protocolVersion: 1 }
    receive(msg)
  }
  const syncSubmitMs = performance.now() - t0
  return { syncSubmitMs, syncOpsPerSec: SYNC_OPS / (syncSubmitMs / 1000) }
}

async function main() {
  console.log('database-core benchmark — TypeScript lane')
  console.log(`  count=${COUNT} sync_ops=${SYNC_OPS}`)

  const dir = join(tmpdir(), `polypack-bench-ts-${Date.now()}`)
  mkdirSync(dir, { recursive: true })

  const { writeMs, writeOpsPerSec } = await benchDurableWrites(dir)
  console.log(`  durable writes: ${writeMs.toFixed(1)}ms (${writeOpsPerSec.toFixed(0)} ops/sec)`)

  const { mutationCount, mutationReplayMs, recoveryMs } = await benchMutationLogAndRecovery(dir)
  console.log(`  mutation log: ${mutationCount} records, replay ${mutationReplayMs.toFixed(2)}ms, recovery (load) ${recoveryMs.toFixed(2)}ms`)

  const { syncSubmitMs, syncOpsPerSec } = benchSyncThroughput()
  console.log(`  sync throughput: ${syncSubmitMs.toFixed(1)}ms (${syncOpsPerSec.toFixed(0)} ops/sec)`)

  rmSync(dir, { recursive: true, force: true })

  mkdirSync(OUT_DIR, { recursive: true })
  const result = {
    results: [{
      lang: 'ts',
      count: COUNT,
      syncOps: SYNC_OPS,
      writeMs,
      writeOpsPerSec,
      mutationCount,
      mutationReplayMs,
      recoveryMs,
      syncSubmitMs,
      syncOpsPerSec,
    }],
  }
  writeFileSync(OUT, JSON.stringify(result, null, 2))
  console.log(`Wrote ${OUT}`)
}

main().catch((err) => {
  console.error(err)
  process.exit(1)
})

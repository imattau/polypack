/** Compare the TypeScript, Rust, and Python durable batch benchmark results. */
import { existsSync, readFileSync, writeFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

type Case = {
  batchSize: number; count: number; mutationCount: number; writeMs: number; writeOpsPerSec: number; compactMs: number
  verified: boolean; nodeCount: number
  before: { walBytes: number; snapshotBytes: number; mutationLogBytes: number }
  after: { walBytes: number; snapshotBytes: number; mutationLogBytes: number }
}
type Result = { schemaVersion: number; lang: string; count: number; seed: number; batchSizes: number[]; cases: Case[] }

const DIR = dirname(fileURLToPath(import.meta.url))
const RESULTS = join(DIR, 'results')
function arg(flag: string, fallback: string): string {
  const i = process.argv.indexOf(flag)
  return i >= 0 && process.argv[i + 1] ? process.argv[i + 1] : fallback
}
function load(path: string): Result {
  if (!existsSync(path)) throw new Error(`missing ${path}`)
  return JSON.parse(readFileSync(path, 'utf8')) as Result
}
const inputs = [
  load(arg('--ts', join(RESULTS, 'database-core-batches-ts.json'))),
  load(arg('--rust', join(RESULTS, 'database-core-batches-rust.json'))),
  load(arg('--python', join(RESULTS, 'database-core-batches-python.json'))),
]
const count = inputs[0].count
const batchSizes = inputs[0].batchSizes
const errors: string[] = []
for (const result of inputs) {
  if (result.schemaVersion !== 1) errors.push(`${result.lang}: unsupported schema version`)
  if (result.count !== count) errors.push(`${result.lang}: count ${result.count} != ${count}`)
  if (result.seed !== inputs[0].seed) errors.push(`${result.lang}: seed mismatch`)
  for (const batch of batchSizes) {
    const c = result.cases.find(item => item.batchSize === batch)
    const expectedMutations = Math.ceil(count / batch)
    if (!c) errors.push(`${result.lang}: missing batch ${batch}`)
    else {
      if (c.mutationCount !== expectedMutations) errors.push(`${result.lang}: batch ${batch} has ${c.mutationCount} mutations, expected ${expectedMutations}`)
      if (!c.verified || c.nodeCount !== count) errors.push(`${result.lang}: batch ${batch} failed verification`)
    }
  }
}

const lines = [
  '# database-core durable batch benchmark',
  '',
  `Workload: ${count} nodes, seed ${inputs[0].seed}. Automatic compaction was disabled; each case explicitly checkpoints once after writes.`,
  '',
  '| binding | batch | write ops/s | write ms | mutations | WAL before | snapshot after | compact ms |',
  '|---|---:|---:|---:|---:|---:|---:|---:|',
]
for (const batch of batchSizes) {
  for (const result of inputs) {
    const c = result.cases.find(item => item.batchSize === batch)!
    lines.push(`| ${result.lang} | ${batch} | ${c.writeOpsPerSec.toFixed(0)} | ${c.writeMs.toFixed(1)} | ${c.mutationCount} | ${c.before.walBytes} | ${c.after.snapshotBytes} | ${c.compactMs.toFixed(2)} |`)
  }
}
lines.push('', '## Interpretation', '',
  'All three bindings flush only dirty records per batch. WAL and mutation-log byte counts still differ where the storage formats differ, but mutation counts and verified graph contents are directly comparable.',
  '', errors.length ? '## Parity errors' : '## Parity', '', errors.length ? errors.map(error => `- ${error}`).join('\n') : 'All bindings produced the expected mutation counts and verified node counts.')
const out = arg('--out', join(DIR, 'database-core-batches-report.md'))
writeFileSync(out, `${lines.join('\n')}\n`)
console.log(lines.join('\n'))
console.log(`Wrote ${out}`)
if (errors.length) process.exitCode = 1

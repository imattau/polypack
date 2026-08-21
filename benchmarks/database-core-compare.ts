/**
 * Merge TypeScript / Rust / Python database-core benchmark JSON into one
 * comparison report.
 *
 * Usage:
 *   npx tsx benchmarks/database-core-compare.ts
 *     [--ts database-core-ts.json] [--rust database-core-rust.json] [--python database-core-python.json]
 *
 * Writes benchmarks/database-core-report.md.
 */
import { readFileSync, writeFileSync, existsSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'

const DIR = dirname(fileURLToPath(import.meta.url))
const RESULTS = join(DIR, 'results')

interface DbCoreResult {
  lang: string
  count: number
  syncOps: number
  writeMs: number
  writeOpsPerSec: number
  mutationCount: number
  mutationReplayMs: number
  recoveryMs: number
  syncSubmitMs: number
  syncOpsPerSec: number
}

function argValue(flag: string): string | null {
  const eq = process.argv.find(a => a.startsWith(`${flag}=`))
  if (eq) return eq.slice(flag.length + 1)
  const i = process.argv.indexOf(flag)
  if (i >= 0 && i + 1 < process.argv.length) return process.argv[i + 1]
  return null
}

function load(name: string): DbCoreResult | null {
  const file = join(RESULTS, name)
  if (!existsSync(file)) return null
  const payload = JSON.parse(readFileSync(file, 'utf8')) as { results: DbCoreResult[] }
  return payload.results[0] ?? null
}

const ts = load(argValue('--ts') ?? 'database-core-ts.json')
const rust = load(argValue('--rust') ?? 'database-core-rust.json')
const python = load(argValue('--python') ?? 'database-core-python.json')

const lanes: Array<{ label: string; r: DbCoreResult | null }> = [
  { label: 'TypeScript', r: ts },
  { label: 'Rust', r: rust },
  { label: 'Python', r: python },
]

const present = lanes.filter(l => l.r !== null) as Array<{ label: string; r: DbCoreResult }>
if (present.length === 0) {
  console.error('No benchmark results found. Run database-core-ts.ts, the Rust example, and bench_db.py first.')
  process.exit(1)
}

function row(label: string, get: (r: DbCoreResult) => string): string {
  return `| ${label} | ${present.map(l => get(l.r)).join(' | ')} |`
}

const header = `| metric | ${present.map(l => l.label).join(' | ')} |`
const sep = `|---${'|---'.repeat(present.length)}|`

const rows = [
  row('durable write throughput (ops/sec)', r => r.writeOpsPerSec.toFixed(0)),
  row('durable write time', r => `${r.writeMs.toFixed(1)}ms`),
  row('mutation log records', r => `${r.mutationCount}`),
  row('mutation log replay time', r => `${r.mutationReplayMs.toFixed(2)}ms`),
  row('cold-store recovery time', r => `${r.recoveryMs.toFixed(2)}ms`),
  row('sync throughput (ops/sec)', r => r.syncOpsPerSec.toFixed(0)),
  row('sync submit time', r => `${r.syncSubmitMs.toFixed(1)}ms`),
]

function speedupTable(): string {
  if (!ts) return 'No TypeScript baseline available for speedup comparison.'
  const lines = ['| lane | write speedup vs TS | sync speedup vs TS |', '|------|----------------------|---------------------|']
  for (const { label, r } of present) {
    if (label === 'TypeScript') continue
    const writeSpeedup = (r.writeOpsPerSec / ts.writeOpsPerSec).toFixed(2)
    const syncSpeedup = (r.syncOpsPerSec / ts.syncOpsPerSec).toFixed(2)
    lines.push(`| ${label} | ${writeSpeedup}× | ${syncSpeedup}× |`)
  }
  return lines.join('\n')
}

const missing = lanes.filter(l => l.r === null).map(l => l.label)
const missingNote = missing.length > 0
  ? `\nMissing lanes: ${missing.join(', ')} — run their benchmark scripts to fill in the comparison.\n`
  : ''

const report = `# database-core benchmark report

Generated ${new Date().toISOString()} by \`benchmarks/database-core-compare.ts\`.

Measures durable write throughput, mutation-log replay + cold-store recovery,
and in-process sync-server throughput, for the same workload
(count=${present[0].r.count} nodes, ${present[0].r.syncOps} sync ops) across
the TypeScript, Rust, and Python implementations of database-core.

- **Durable write throughput**: \`addNode\`/\`add_node\` in batches of 500,
  flushed to a real on-disk store (\`BinaryStoreAdapter\` / \`FileStorage\` /
  native \`DirectoryStorage\`).
- **Mutation log / recovery**: the store is reopened cold; \`mutation log
  replay time\` reads back the durable mutation log, \`cold-store recovery
  time\` is how long re-opening + loading the store into the working set
  takes (\`PolyGraph.load()\` / \`Graph::warm()\` / \`open_store()\`).
- **Sync throughput**: ops submitted directly to each language's
  \`SyncServer\` message handler (no transport/network layer), in batches of
  100.
${missingNote}
## Results

${header}
${sep}
${rows.join('\n')}

## Speedups relative to TypeScript

${speedupTable()}

## Notes

- Rust batches durable writes every 500 nodes via \`add_nodes\` +
  \`flush()\`; TypeScript and Python do the same via per-node \`addNode\`/
  \`add_node\` + a \`flush()\`/\`save()\` every 500 nodes — so mutation-log
  record counts line up across lanes for the same workload.
- Python's numbers include the pure-Python graph layer + native-extension
  FFI overhead per node; only the vector/index primitives are native.
- These are single-run, single-machine numbers meant to catch regressions
  and give a rough cross-language comparison — not a rigorous statistical
  benchmark (no warmup, no repeated trials, no percentiles).
`

writeFileSync(join(DIR, 'database-core-report.md'), report)
console.log(`Wrote ${join(DIR, 'database-core-report.md')}`)

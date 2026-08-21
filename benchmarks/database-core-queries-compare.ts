/** Compare cross-binding graph-query and vector benchmark results. */
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const root = dirname(fileURLToPath(import.meta.url))
const files = ['database-core-queries-ts.json', 'database-core-queries-rust.json', 'database-core-queries-python.json']
const rows = files.map(file => JSON.parse(readFileSync(join(root, 'results', file), 'utf8')))
const field = (r: any, name: string, key = 'p50Ms') => Number(r[name]?.[key] ?? 0)
const lines = [
  '# database-core read benchmark', '',
  `Generated ${new Date().toISOString()}. Workload: ${rows[0].count.toLocaleString()} seeded random vectors (${rows[0].dimensions} dimensions, data seed ${rows[0].dataSeed}, query seed ${rows[0].querySeed}), ${rows[0].iterations} measured queries. HNSW efSearch=300.`, '',
  'This compares the common hot graph-query and vector-index APIs. Python currently does not expose the persisted-query builder, so persisted-query latency is intentionally not mixed into this cross-binding table.', '',
  '| binding | graph query p50 | exact vector p50 | HNSW p50 | Recall@10 |', '|---|---:|---:|---:|---:|',
  ...rows.map(r => `| ${r.lang} | ${field(r, 'graphQuery').toFixed(3)}ms | ${field(r, 'exactVector').toFixed(3)}ms | ${field(r, 'hnswVector').toFixed(3)}ms | ${(Number(r.hnswVector.recallAtK) * 100).toFixed(1)}% |`), '',
  '## Verification', '',
  ...rows.map(r => `- ${r.lang}: graph query ${r.graphQuery.resultCount} rows; exact ${r.exactVector.resultCount}; HNSW ${r.hnswVector.resultCount}.`), '',
  'These are single-process measurements intended for regression detection. Repeat runs are recommended before drawing performance conclusions.',
]
const out = join(root, 'database-core-queries-report.md'); mkdirSync(dirname(out), { recursive: true }); writeFileSync(out, lines.join('\n')); console.log(lines.join('\n')); console.log(`Wrote ${out}`)

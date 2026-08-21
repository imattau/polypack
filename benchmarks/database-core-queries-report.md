# database-core read benchmark

Generated 2026-08-21T11:02:02.621Z. Workload: 10,000 seeded random vectors (32 dimensions, data seed 42, query seed 43), 20 measured queries. HNSW efSearch=300.

This compares the common hot graph-query and vector-index APIs. Python currently does not expose the persisted-query builder, so persisted-query latency is intentionally not mixed into this cross-binding table.

| binding | graph query p50 | exact vector p50 | HNSW p50 | Recall@10 |
|---|---:|---:|---:|---:|
| ts | 0.212ms | 0.549ms | 1.772ms | 100.0% |
| rust | 0.346ms | 0.204ms | 1.201ms | 100.0% |
| python | 95.923ms | 0.214ms | 1.195ms | 100.0% |

## Verification

- ts: graph query 25 rows; exact 10; HNSW 10.
- rust: graph query 25 rows; exact 10; HNSW 10.
- python: graph query 25 rows; exact 10; HNSW 10.

These are single-process measurements intended for regression detection. Repeat runs are recommended before drawing performance conclusions.
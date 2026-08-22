# database-core mixed read/write benchmark

Workload: 2000 initial nodes, 200 rounds. Each round performs write, update, hot query, persisted query, and vector query in a deterministic single-graph schedule.

| binding | total ms | write p50 | update p50 | hot query p50 | persisted p50 | vector p50 |
|---|---:|---:|---:|---:|---:|---:|
| ts | 317.4 | 0.074ms | 0.045ms | 0.240ms | 0.331ms | 0.357ms |
| rust | 1568.3 | 1.161ms | 0.451ms | 0.142ms | 0.341ms | 0.626ms |
| python | 932.5 | 1.352ms | 1.323ms | 0.196ms | 0.437ms | 0.075ms |

## Verification

- ts: final node count 2200; all operation counts verified
- rust: final node count 2200; all operation counts verified
- python: final node count 2200; all operation counts verified

This is a cooperative mixed workload over one graph instance, not a multi-writer thread-safety test. Results are intended for regression detection; repeat runs for stable comparisons.
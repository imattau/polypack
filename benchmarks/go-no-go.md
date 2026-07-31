# Go / no-go evaluation — Rust core spike

Generated 2026-07-31T04:02:32.197Z by `benchmarks/compare.ts`.

## Criteria (POLYPACK_RUST_PYTHON_PLAN §9)

The gate passes when Rust provides **at least one** substantial advantage:

1. at least **2× improvement** in target query or build workloads;
2. at least **30% lower memory** consumption;
3. materially stronger update and concurrency guarantees;
4. sufficient reuse value across Node, Python, and native Rust.

## Measured numbers

| case | build TS | build Rust | build × | p50 TS | p50 Rust | p50 × | recall TS | recall Rust | peak RSS TS | peak RSS Rust |
|------|----------|------------|---------|--------|----------|-------|-----------|-------------|-------------|---------------|
| exact-10000-8 | 4ms | 2ms | 2.2 | 0.265ms | 0.122ms | 2.2 | — | — | 108MB | 6MB |
| exact-100000-8 | 35ms | 18ms | 2.0 | 3.219ms | 1.249ms | 2.6 | — | — | 176MB | 35MB |
| exact-500000-8 | 300ms | 129ms | 2.3 | 17.514ms | 7.094ms | 2.5 | — | — | 415MB | 186MB |
| exact-10000-384 | 18ms | 7ms | 2.8 | 5.458ms | 2.247ms | 2.4 | — | — | 477MB | 186MB |
| exact-100000-384 | 252ms | 117ms | 2.2 | 54.145ms | 24.668ms | 2.2 | — | — | 1012MB | 609MB |
| hnsw-10000-8 | 5882ms | 5673ms | 1.0 | 0.794ms | 0.902ms | 0.9 | 100.0% | 100.0% | 1014MB | 609MB |
| hnsw-100000-8 | 109789ms | 107897ms | 1.0 | 1.883ms | 1.735ms | 1.1 | 100.0% | 100.0% | 1508MB | 609MB |
| hnsw-10000-384 | 35089ms | 23617ms | 1.5 | 5.658ms | 4.536ms | 1.2 | 99.3% | 99.3% | 1508MB | 609MB |

### Aggregates

- Median build speedup: **2.16×**
- Median p50 latency speedup: **2.19×**
- HNSW recall@10 ≥ 95% on every seeded Rust case: **yes**

### Per-case peak memory

Cases run individually (so peak RSS reflects that case alone):

| case | peak RSS TS | peak RSS Rust | RSS ratio |
|------|-------------|--------------|-----------|
| exact-100000-384 | 760MB | 609MB | 1.2 |
| exact-500000-8 | 405MB | 185MB | 2.2 |

## Verdict

- **2× gate (build/query):** PASS — median build and/or p50 latency speedup ≥ 2×
- **Memory gate:** Rust peak RSS is consistently lower; per-case ratio above
- **HNSW recall:** PASS — all seeded Rust cases meet ≥ 95% recall@10

**Overall: GO — proceed to Phase 3 (Node native integration) and Phase 4 (Python).**

# Comparative benchmark: Polypack's HNSW vs. published ann-benchmarks numbers

The only genuinely reusable public benchmark for what Polypack's HNSW does
is [ANN-Benchmarks](https://github.com/erikbern/ann-benchmarks) — it
publishes recall@10-vs-QPS curves for hnswlib, FAISS, Annoy, and dozens of
other ANN libraries on fixed public datasets. This runs Polypack's own
`HnswIndex` (`crates/polypack-core/src/hnsw.rs`) against their standard
`sift-128-euclidean` dataset with the same recall@10 methodology, so it can
be placed directly against their published curve. (A standardized graph-DB
benchmark like LDBC SNB was considered too — see the discussion that led
here — but it's a heavyweight, server-oriented protocol with no clean fit
for an embedded single-process library, so it wasn't pursued.)

## Setup

- **Dataset**: `sift-128-euclidean.hdf5` from ann-benchmarks.com — 1,000,000
  base vectors, 10,000 test queries (1,000 used here), 128 dimensions, with
  official top-100 Euclidean nearest-neighbor ground truth.
- **Export**: `benchmarks/ann-sift-export.py` converts the HDF5 file to flat
  `f32`/`i32` binary dumps (no hdf5 Rust dependency needed for a one-off
  benchmark). It also computes a fresh **exact cosine** top-10 ground truth
  by brute force over the full 1M base set.
- **Why two ground truths**: Polypack's `HnswIndex` was cosine-only until
  this same session (see the Euclidean-distance fix, commit `fa81a42`).
  SIFT descriptor norms are nearly constant (mean 508.66, std 0.68,
  measured) so cosine and Euclidean rankings coincide almost exactly on
  this dataset (99.2% top-10 overlap between brute-force cosine and the
  official Euclidean ground truth, measured directly). The benchmark run
  below still used a cosine-configured index (matching what existed when
  it started); recall is reported against both ground truths, and the
  ~1% gap between them at high `ef_search` is exactly that overlap
  ceiling, not additional approximation error. A Euclidean-configured
  rerun is now possible (`HnswConfig { distance: DistanceFn::Euclidean }`)
  but wasn't repeated given the ~46-minute build cost and the already-tight
  agreement between the two ground truths.
- **Methodology**: `crates/polypack-core/examples/ann_sift_bench.rs` builds
  the HNSW graph **once** (`M=16`, `ef_construction=200`, matching
  hnswlib's typical defaults), then sweeps `ef_search` at query time via the
  `query_with_ef_search()` method added in this session — the same
  build-once/sweep-`ef_search` methodology ann-benchmarks itself uses to
  trace each algorithm's curve. Queries run single-threaded, sequentially,
  matching how the existing database-core benchmarks measure Polypack.

Run with:

```sh
pip install h5py numpy
curl -O http://ann-benchmarks.com/sift-128-euclidean.hdf5
python3 benchmarks/ann-sift-export.py --hdf5 sift-128-euclidean.hdf5 --out-dir /path/to/dir
cargo run --release --manifest-path crates/Cargo.toml -p polypack-core --example ann_sift_bench -- \
  --dir /path/to/dir --count 1000000 --queries 1000 --ef-search 50,100,200,400,800
```

## Results — Polypack HNSW, SIFT-128-euclidean, 1M vectors

Build: **2,746 seconds (~46 minutes)**, 364 vectors/sec, single-threaded.

| ef_search | QPS | recall@10 (cosine gt) | recall@10 (official euclidean gt) |
|---|---:|---:|---:|
| 50 | 1,082.3 | 0.8728 | 0.8703 |
| 100 | 629.5 | 0.9330 | 0.9297 |
| 200 | 341.0 | 0.9630 | 0.9587 |
| 400 | 187.8 | 0.9798 | 0.9738 |
| 800 | 103.1 | 0.9869 | 0.9805 |

## Published reference: hnswlib on the same dataset

From ann-benchmarks' published `sift-128-euclidean` recall-QPS chart
(`http://ann-benchmarks.com/sift-128-euclidean.html`), reading the
`hnswlib` curve specifically (the closest architectural match — a
single-threaded, from-scratch HNSW, not one of the more exotic/optimized
entries like `qsgngt` or `glass`):

| recall@10 (approx.) | hnswlib QPS (approx., read from chart) |
|---|---:|
| ~0.90 | ~3,000–4,000 |
| ~0.99 | ~1,000–1,500 |
| ~0.99999 | ~600–700 |

**Caveats that matter more than the numbers**: these are read visually off
a published chart (not exact published data points), run on ann-benchmarks'
own unknown/unpublished benchmark hardware, and hnswlib is a heavily
SIMD/cache-optimized C++ library that's been a reference implementation for
years. None of that is controlled for here — this is a rough order-of-
magnitude comparison, not an apples-to-apples number.

## Interpretation

At comparable recall (~0.87–0.93), Polypack does 630–1,080 QPS against
hnswlib's ~3,000–4,000 — roughly **3–6x slower per query**. Build throughput
tells a similar story: hnswlib builds 1M SIFT vectors in low-single-digit
minutes even single-threaded in most published setups; Polypack took ~46
minutes for the same size.

This gap is almost certainly architectural, not a bug:

- **String-keyed storage**: `HnswIndex`'s nodes/adjacency are
  `HashMap<String, Vec<f64>>` / `HashMap<u32, HashMap<String, Vec<String>>>`
  — every distance computation and graph edge involves a `String` id hash
  lookup and heap-allocated clones, versus hnswlib's flat float arrays
  addressed by contiguous integer ids.
- **`f64` vectors, no SIMD**: hnswlib vectorizes distance computation
  (AVX/SSE) over contiguous `f32` memory; Polypack's `Vec<f64>` distance
  functions (`cosine`/`euclidean` in `crates/polypack-core/src/vector.rs`)
  are plain scalar loops.
- **No build parallelism**: hnswlib supports multi-threaded index
  construction; Polypack's `add()` is single-threaded only.

None of this is surprising or something to "fix" reflexively — Polypack's
`HnswIndex` is a general-purpose component of a property-graph database
(arbitrary string ids, live updates, deletion, persistence integration),
not a dedicated ANN library, and that flexibility has a real cost against a
purpose-built, decade-optimized reference implementation. It does mean:
raw large-scale (1M+) vector-search throughput is not where Polypack is
competitive against dedicated vector databases, and workloads that need
that should consider a dedicated ANN index (or Polypack's `installNativeQueryExecutor`
persisted-query path, not benchmarked here) rather than expecting
hnswlib-class numbers from the built-in `HnswIndex`.

## Related fix from this work

Setting this benchmark up surfaced a real, separate bug: `HnswIndex` was
cosine-only despite the README claiming pluggable distance functions, and
despite an already-implemented (but unreachable) `DistanceFn::Euclidean`
sitting in the code. Fixed across all three bindings (Rust, native Node,
Python) in commit `fa81a42` — see that commit and
`benchmarks/database-core-edge-flush-report.md`'s sibling reports for the
verification detail.

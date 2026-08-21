# Scale characterization

`scale-characterization.ts` measures the full-store and hot-working-set
behaviour of the TypeScript embedded database at 100K, 1M, and 10M nodes:

- insert throughput and update throughput;
- persisted (cold) and hot graph queries;
- persisted and hot traversal;
- store startup, warm time, mutation-log replay, and storage bytes;
- process heap/RSS;
- exact-vector and HNSW build/query latency.

Run the default matrix with:

```sh
npm run bench:scale
```

The default vector scope is capped at 100K nodes so the 10M-node storage and
graph run remains a practical experiment. The report records this explicitly
as `vectorScope: "capped"`. To characterize vector indexes over the full
dataset, opt in on a machine sized for that workload:

```sh
npm run bench:scale -- --vector-count 0
```

Use `--sizes 100000` or `--sizes 100000,1000000` for staged runs. Results are
written to `benchmarks/results/scale-characterization.json`.

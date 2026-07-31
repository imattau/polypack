# In-memory query delegation gate

Generated 2026-07-31T06:21:49.545Z. Graph: 50,000 nodes
(16666.666666666668 of type 'doc'), query: whereNodeType('doc') + orderBy(score desc)
+ limit(20), 50 iterations.

| engine | p50 | p95 | p99 |
|--------|-----|-----|-----|
| TypeScript | 4.378ms | 5.276ms | 7.690ms |
| native delegation | 201.307ms | 211.957ms | 224.887ms |

## Verdict

Native delegation is slower on p50 (serialization overhead) — keep in-memory GraphQuery on TypeScript.

**Recommendation (in-memory GraphQuery):** stay on the TypeScript pipeline. The
native executor is exposed for scenarios where the per-call serialization is
amortised:

- **Python** GraphQuery delegates to the Rust executor (Python's per-node
  interpreter overhead makes the batch path a win despite conversion cost).
- **Whole-store / persisted queries** where nodes are already serialized and
  the query cost (similarity over many vectors, deep traversal) dominates.
- Opt-in via installNativeQueryExecutor(); it is not installed by default.

engineInfo: {"graph":"typescript","vector":"rust-native","storage":"host","available":true,"query":"rust-native"}

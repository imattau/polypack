"""Cross-binding read benchmark — Python lane."""
from __future__ import annotations
import argparse, json, math, time
from pathlib import Path
from . import PolyGraph, ExactIndex, HnswIndex

def mulberry32(seed: int):
    state = seed & 0xFFFFFFFF
    def next_f64():
        nonlocal state
        state = (state + 0x6D2B79F5) & 0xFFFFFFFF
        t = (((state ^ (state >> 15)) * (1 | state)) & 0xFFFFFFFF)
        t = ((t + (((t ^ (t >> 7)) * (61 | t)) & 0xFFFFFFFF)) ^ t) & 0xFFFFFFFF
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296.0
    return next_f64
def vectors(count: int, seed: int, dims: int = 32):
    rng = mulberry32(seed)
    return [[rng() * 2.0 - 1.0 for _ in range(dims)] for _ in range(count)]
def node(i: int, value): return {"id": f"n{i}", "type": ["user", "post", "comment"][i % 3], "data": {"score": i % 1000, "bucket": i % 10, "value": i}, "vector": value, "insertedAt": i, "updatedAt": i}
def measure(fn, iterations):
    times = []; value = None
    for _ in range(iterations):
        start = time.perf_counter(); value = fn(); times.append((time.perf_counter() - start) * 1000)
    ordered = sorted(times)
    at = lambda p: ordered[min(len(ordered) - 1, max(0, math.ceil(len(ordered) * p) - 1))]
    return {"p50Ms": at(.5), "p95Ms": at(.95), "p99Ms": at(.99), "value": value}
def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--count", type=int, default=10000); parser.add_argument("--iterations", type=int, default=20); parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[2] / "benchmarks/results/database-core-queries-python.json"); args = parser.parse_args()
    data = vectors(args.count, 42)
    graph = PolyGraph()
    for i in range(args.count): graph.add_node(node(i, data[i]))
    expected = min(25, 0 if args.count <= 22 else (args.count - 23) // 30 + 1)
    persisted = measure(lambda: graph.query().where_type("post").where("bucket", 2).order_by("score", "desc").limit(25).to_list(), args.iterations)
    hot = measure(lambda: graph.query().where_type("post").where("bucket", 2).order_by("score", "desc").limit(25).ids(), args.iterations)
    if len(persisted["value"]) != expected or len(hot["value"]) != expected: raise RuntimeError("query result verification failed")
    q = vectors(1, 43)[0]; exact = ExactIndex(); exact.add_many((f"n{i}", data[i]) for i in range(args.count)); exact_result = measure(lambda: exact.query(q, 10), args.iterations)
    hnsw = HnswIndex(ef_search=300, level_seed=7); hnsw.add_many((f"n{i}", data[i]) for i in range(args.count)); hnsw_result = measure(lambda: hnsw.query(q, 10), args.iterations)
    if len(exact_result["value"]) != 10 or len(hnsw_result["value"]) != 10: raise RuntimeError("vector result verification failed")
    exact_ids = {x[0] if isinstance(x, tuple) else x["id"] for x in exact_result["value"]}; recall = sum((x[0] if isinstance(x, tuple) else x["id"]) in exact_ids for x in hnsw_result["value"]) / 10
    def clean(x, result_count): return {k: v for k, v in x.items() if k != "value"} | {"resultCount": result_count}
    result = {"schemaVersion": 1, "lang": "python", "count": args.count, "dimensions": 32, "iterations": args.iterations, "topK": 10, "dataSeed": 42, "querySeed": 43, "graphQuery": clean(persisted, len(persisted["value"])), "hotQuery": clean(hot, len(hot["value"])), "exactVector": clean(exact_result, 10), "hnswVector": clean(hnsw_result, 10) | {"recallAtK": recall}}
    args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(result, indent=2)); print(json.dumps(result, indent=2)); print(f"Wrote {args.out}")
if __name__ == "__main__": main()

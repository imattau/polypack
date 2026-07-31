"""Reproducible Python benchmark mirroring benchmarks/run-ts.ts.

Generates the same seeded datasets (fresh mulberry32(42) per case, id-major
then dim-minor) and reports build time, query-latency percentiles, and
Recall@10 against the native exact/HNSW indexes.

Build the extension in release mode first for meaningful numbers:
    maturin develop --release

Usage: python -m polypack.bench [--out benchmarks/results/python-all.json]
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from . import ExactIndex, HnswIndex

TOP_K = 10
HNSW_ARGS = {"m": 16, "mmax0": 32, "ef_construction": 200, "ef_search": 300}

_MASK32 = 0xFFFFFFFF


def _int32(x: int) -> int:
    x &= _MASK32
    return x - 0x100000000 if x >= 0x80000000 else x


def mulberry32(seed: int):
    """Bit-for-bit match with the JS mulberry32 used by the TS benchmark."""
    state = seed & _MASK32

    def next_f64() -> float:
        nonlocal state
        state = _int32(state + 0x6D2B79F5)
        a = state
        t = _int32(_int32(a ^ ((a & _MASK32) >> 15)) * (1 | a))
        t = _int32((t + _int32(_int32(t ^ ((t & _MASK32) >> 7)) * (61 | t))) ^ t)
        return ((t ^ ((t & _MASK32) >> 14)) & _MASK32) / 4294967296.0

    return next_f64


def _percentile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    idx = min(len(sorted_values) - 1, max(0, int((p / 100.0) * len(sorted_values)) + 1 - 1))
    return sorted_values[idx]


def _generate(count: int, dims: int) -> list[list[float]]:
    rng = mulberry32(42)
    return [[rng() * 2.0 - 1.0 for _ in range(dims)] for _ in range(count)]


def _run_case(index: str, count: int, dims: int, queries: int) -> dict:
    data = _generate(count, dims)
    query_vectors = _generate(queries, dims)

    exact = ExactIndex("cosine")
    t0 = time.perf_counter()
    exact.add_many((f"v{i}", v) for i, v in enumerate(data))
    exact_build = (time.perf_counter() - t0) * 1000.0

    hnsw = HnswIndex(**HNSW_ARGS)
    t0 = time.perf_counter()
    if index == "hnsw":
        hnsw.add_many((f"v{i}", v) for i, v in enumerate(data))
    hnsw_build = (time.perf_counter() - t0) * 1000.0

    if index == "exact":
        lat = []
        for q in query_vectors:
            t0 = time.perf_counter()
            exact.query(q, TOP_K, 0.0)
            lat.append((time.perf_counter() - t0) * 1000.0)
        lat.sort()
        return {
            "name": f"exact-{count}-{dims}",
            "index": "exact",
            "count": count,
            "dims": dims,
            "buildMs": exact_build,
            "queryCount": queries,
            "avgMs": sum(lat) / len(lat),
            "p50": _percentile(lat, 50),
            "p95": _percentile(lat, 95),
            "p99": _percentile(lat, 99),
            "recall10": None,
        }

    lat = []
    hits = 0
    for q in query_vectors:
        exact_ids = {r[0] for r in exact.query(q, TOP_K, 0.0)}
        t0 = time.perf_counter()
        ann = hnsw.query(q, TOP_K, 0.0)
        lat.append((time.perf_counter() - t0) * 1000.0)
        hits += sum(1 for r in ann if r[0] in exact_ids)
    lat.sort()
    return {
        "name": f"hnsw-{count}-{dims}",
        "index": "hnsw",
        "count": count,
        "dims": dims,
        "buildMs": hnsw_build,
        "queryCount": queries,
        "avgMs": sum(lat) / len(lat),
        "p50": _percentile(lat, 50),
        "p95": _percentile(lat, 95),
        "p99": _percentile(lat, 99),
        "recall10": hits / (queries * TOP_K),
    }


CASES = [
    ("exact", 10_000, 8, 1000),
    ("exact", 100_000, 8, 1000),
    ("exact", 500_000, 8, 500),
    ("exact", 10_000, 384, 500),
    ("exact", 100_000, 384, 200),
    ("hnsw", 10_000, 8, 1000),
    ("hnsw", 100_000, 8, 1000),
    ("hnsw", 10_000, 384, 500),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    results = []
    for index, count, dims, queries in CASES:
        r = _run_case(index, count, dims, queries)
        results.append(r)
        print(f"  {r['name']} build={r['buildMs']:.0f}ms recall={r['recall10']} p50={r['p50']:.3f}ms")

    payload = {"engine": "python", "hnswConfig": HNSW_ARGS, "topK": TOP_K, "results": results}
    text = json.dumps(payload, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text)
    else:
        print(text)


if __name__ == "__main__":
    main()

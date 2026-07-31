"""100K-vector example — verifies a wheel install works from a clean venv.

Run against the built wheel (no Rust toolchain required):

    pip install polypack-0.1.0-*.whl
    python examples/100k.py
"""

from __future__ import annotations

import time

from polypack import HnswIndex, PolyGraph, engine_info
from polypack.bench import mulberry32


def main() -> None:
    count = 100_000
    dims = 8
    print(f"engine: {engine_info()}")
    print(f"building {count} x {dims}-dim vectors (seeded) ...")

    graph = PolyGraph()
    rng = mulberry32(42)
    t0 = time.perf_counter()
    for i in range(count):
        vector = [rng() * 2.0 - 1.0 for _ in range(dims)]
        graph.add_node(
            {"id": f"n{i}", "type": "doc", "data": {"i": i}, "vector": vector, "insertedAt": i, "updatedAt": i}
        )
    build_ms = (time.perf_counter() - t0) * 1000.0
    print(f"insert: {build_ms:.0f}ms ({count / (build_ms / 1000):.0f} n/s)")

    query = [1.0] + [0.0] * (dims - 1)
    t0 = time.perf_counter()
    results = graph.query().similar_to(query, 0.5, 5).ids()
    exact_ms = (time.perf_counter() - t0) * 1000.0
    print(f"exact similar_to top-5: {results} in {exact_ms:.1f}ms")

    hnsw = HnswIndex(ef_search=300)
    rng = mulberry32(42)
    t0 = time.perf_counter()
    hnsw.add_many((f"n{i}", [rng() * 2.0 - 1.0 for _ in range(dims)]) for i in range(count))
    hnsw_ms = (time.perf_counter() - t0) * 1000.0
    print(f"hnsw build: {hnsw_ms:.0f}ms ({count / (hnsw_ms / 1000):.0f} n/s)")

    t0 = time.perf_counter()
    ann = hnsw.query(query, 5)
    ann_ms = (time.perf_counter() - t0) * 1000.0
    print(f"hnsw top-5: {[r[0] for r in ann]} in {ann_ms:.2f}ms")

    assert graph.size == count
    assert len(hnsw) == count
    print("OK — 100K-vector example passed")


if __name__ == "__main__":
    main()

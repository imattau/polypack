"""Cross-binding durable batch-size benchmark — Python lane."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path

from . import PolyGraph

BATCHES = (1, 100, 500, 5000)
NO_AUTO_COMPACT = 1_000_000_000
MASK32 = 0xFFFFFFFF


def mulberry32(seed: int):
    state = seed & MASK32

    def next_f64() -> float:
        nonlocal state
        state = (state + 0x6D2B79F5) & MASK32
        a = state
        t = (((a ^ (a >> 15)) * (1 | a)) & MASK32)
        t = ((t + (((t ^ (t >> 7)) * (61 | t)) & MASK32)) ^ t) & MASK32
        return ((t ^ (t >> 14)) & MASK32) / 4294967296.0

    return next_f64


def make_node(i: int, rng) -> dict:
    return {
        "id": f"n{i}",
        "type": ["user", "post", "comment"][i % 3],
        "data": {"idx": i, "value": rng(), "tag": f"tag_{i % 50}"},
        "insertedAt": i,
        "updatedAt": i,
    }


def size(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def run_case(root: Path, count: int, seed: int, batch_size: int) -> dict:
    directory = root / f"batch-{batch_size}"
    graph = PolyGraph()
    graph.open_store(str(directory), compact_threshold=NO_AUTO_COMPACT)
    rng = mulberry32(seed)
    start = time.perf_counter()
    for i in range(count):
        graph.add_node(make_node(i, rng))
        if (i + 1) % batch_size == 0:
            graph.save()
    if count % batch_size:
        graph.save()
    write_ms = (time.perf_counter() - start) * 1000.0
    mutations = len(graph.mutation_log())
    before = {
        "walBytes": size(directory / "wal.msgpack"),
        "snapshotBytes": size(directory / "snapshot.msgpack"),
        "mutationLogBytes": size(directory / "mutations.jsonl"),
    }
    start = time.perf_counter()
    graph.checkpoint()
    compact_ms = (time.perf_counter() - start) * 1000.0
    after = {
        "walBytes": size(directory / "wal.msgpack"),
        "snapshotBytes": size(directory / "snapshot.msgpack"),
        "mutationLogBytes": size(directory / "mutations.jsonl"),
    }
    verification = graph.verify()
    graph.close_store()
    return {
        "batchSize": batch_size,
        "count": count,
        "seed": seed,
        "writeMs": write_ms,
        "writeOpsPerSec": count / (write_ms / 1000.0),
        "mutationCount": mutations,
        "compactMs": compact_ms,
        "verified": bool(verification["ok"]),
        "nodeCount": int(verification["nodeCount"]),
        "before": before,
        "after": after,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[2] / "benchmarks/results/database-core-batches-python.json")
    args = parser.parse_args()
    if args.count < 1:
        raise SystemExit("--count must be positive")
    root = Path(tempfile.mkdtemp(prefix="polypack-batches-py-"))
    try:
        result = {"schemaVersion": 1, "lang": "python", "count": args.count, "seed": args.seed, "batchSizes": list(BATCHES), "cases": [run_case(root, args.count, args.seed, batch) for batch in BATCHES]}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        print(f"Wrote {args.out}")
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()

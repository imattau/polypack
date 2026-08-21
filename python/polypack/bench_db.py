"""database-core benchmark — Python lane.

Companion to `benchmarks/database-core-ts.ts` and
`crates/polypack-graph/examples/database_core_bench.rs`: measures the same
three things against a real on-disk store so
`benchmarks/database-core-compare.ts` can merge all three into one report.

  1. durable write throughput — `PolyGraph.add_node` + `save()` against the
     native store (`open_store`), backed by real files.
  2. mutation-log replay + recovery — reopen the store cold and time
     `mutation_log()` and `open_store()` itself (which eagerly loads state).
  3. sync throughput — `SyncServer.receive()`, fed directly (bypassing any
     transport) to measure raw op-ingestion throughput.

Requires the native extension (`maturin develop --release`).

Usage: python -m polypack.bench_db [--count N] [--sync-ops N] [--out PATH]
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path

from . import PolyGraph
from .sync import SyncServer

_MASK32 = 0xFFFFFFFF


def _int32(x: int) -> int:
    x &= _MASK32
    return x - 0x100000000 if x >= 0x80000000 else x


def mulberry32(seed: int):
    """Bit-for-bit match with the TS/Rust mulberry32."""
    state = seed & _MASK32

    def next_f64() -> float:
        nonlocal state
        state = _int32(state + 0x6D2B79F5)
        a = state
        t = _int32(_int32(a ^ ((a & _MASK32) >> 15)) * (1 | a))
        t = _int32((t + _int32(_int32(t ^ ((t & _MASK32) >> 7)) * (61 | t))) ^ t)
        return ((t ^ ((t & _MASK32) >> 14)) & _MASK32) / 4294967296.0

    return next_f64


def _make_node(i: int, rng) -> dict:
    return {
        "id": f"n{i}",
        "type": ["user", "post", "comment"][i % 3],
        "data": {"idx": i, "value": rng(), "tag": f"tag_{i % 50}"},
        "insertedAt": i,
        "updatedAt": i,
    }


def _bench_durable_writes(directory: str, count: int, seed: int) -> dict:
    graph = PolyGraph()
    graph.open_store(directory)
    rng = mulberry32(seed)
    flush_every = 500
    t0 = time.perf_counter()
    for i in range(count):
        graph.add_node(_make_node(i, rng))
        if (i + 1) % flush_every == 0:
            graph.save()
    if count % flush_every:
        graph.save()
    graph.close_store()
    write_ms = (time.perf_counter() - t0) * 1000.0
    return {"writeMs": write_ms, "writeOpsPerSec": count / (write_ms / 1000.0)}


def _bench_mutation_log_and_recovery(directory: str) -> dict:
    graph = PolyGraph()
    t0 = time.perf_counter()
    graph.open_store(directory)
    recovery_ms = (time.perf_counter() - t0) * 1000.0

    t1 = time.perf_counter()
    mutations = graph.mutation_log()
    replay_ms = (time.perf_counter() - t1) * 1000.0

    graph.close_store()
    return {"mutationCount": len(mutations), "mutationReplayMs": replay_ms, "recoveryMs": recovery_ms}


def _bench_sync_throughput(sync_ops: int) -> dict:
    server = SyncServer()
    client_id = "bench-client"
    receive = server.add_client(client_id, send=lambda _msg: None)
    batch_size = 100
    seq = 0
    t0 = time.perf_counter()
    offset = 0
    while offset < sync_ops:
        end = min(offset + batch_size, sync_ops)
        ops = []
        for _ in range(offset, end):
            seq += 1
            ops.append({
                "seq": seq,
                "timestamp": seq,
                "clientId": client_id,
                "kind": "addNode",
                "payload": {"id": f"s{seq}"},
                "operationId": f"{client_id}:{seq}",
            })
        receive({"type": "delta", "clientId": client_id, "fromSeq": offset, "ops": ops, "protocolVersion": 1})
        offset = end
    submit_ms = (time.perf_counter() - t0) * 1000.0
    return {"syncSubmitMs": submit_ms, "syncOpsPerSec": sync_ops / (submit_ms / 1000.0)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=20_000)
    parser.add_argument("--sync-ops", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    out_path = Path(args.out) if args.out else Path(__file__).resolve().parent.parent.parent / "benchmarks" / "results" / "database-core-python.json"

    print("database-core benchmark — Python lane")
    print(f"  count={args.count} sync_ops={args.sync_ops}")

    directory = tempfile.mkdtemp(prefix="polypack-bench-py-")
    try:
        writes = _bench_durable_writes(directory, args.count, args.seed)
        print(f"  durable writes: {writes['writeMs']:.1f}ms ({writes['writeOpsPerSec']:.0f} ops/sec)")

        recovery = _bench_mutation_log_and_recovery(directory)
        print(f"  mutation log: {recovery['mutationCount']} records, replay {recovery['mutationReplayMs']:.2f}ms, recovery (open_store) {recovery['recoveryMs']:.2f}ms")

        sync = _bench_sync_throughput(args.sync_ops)
        print(f"  sync throughput: {sync['syncSubmitMs']:.1f}ms ({sync['syncOpsPerSec']:.0f} ops/sec)")
    finally:
        shutil.rmtree(directory, ignore_errors=True)

    result = {
        "results": [{
            "lang": "python",
            "count": args.count,
            "syncOps": args.sync_ops,
            **writes,
            **recovery,
            **sync,
        }]
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

"""Deterministic mixed read/write workload — Python lane."""
from __future__ import annotations
import argparse, json, math, shutil, tempfile, time
from pathlib import Path
from . import PolyGraph

def rng(seed):
    state = seed & 0xFFFFFFFF
    def next_f64():
        nonlocal state
        state = (state + 0x6D2B79F5) & 0xFFFFFFFF
        t = ((state ^ (state >> 15)) * (1 | state)) & 0xFFFFFFFF
        t = ((t + (((t ^ (t >> 7)) * (61 | t)) & 0xFFFFFFFF)) ^ t) & 0xFFFFFFFF
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296.0
    return next_f64
def vec(i, dims=32):
    r = rng(i + 42); return [r() * 2 - 1 for _ in range(dims)]
def node(i): return {"id": f"n{i}", "type": "comment" if i % 2 else "post", "data": {"score": i, "bucket": i % 10, "value": i}, "vector": vec(i), "insertedAt": i, "updatedAt": i}
def summary(values):
    s = sorted(values); at = lambda p: s[min(len(s)-1, max(0, math.ceil(len(s)*p)-1))]
    return {"count": len(values), "p50Ms": at(.5), "p95Ms": at(.95), "p99Ms": at(.99), "opsPerSec": len(values) / (sum(values) / 1000)}
def main():
    p = argparse.ArgumentParser(); p.add_argument("--initial", type=int, default=2000); p.add_argument("--rounds", type=int, default=200); p.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[2] / "benchmarks/results/database-core-mixed-python.json"); a=p.parse_args()
    directory = Path(tempfile.mkdtemp(prefix="polypack-mixed-py-")); times = {k: [] for k in ("write", "update", "hotQuery", "persistedQuery", "vectorQuery")}; start=time.perf_counter()
    try:
        g=PolyGraph.open(str(directory))
        for i in range(a.initial): g.add_node(node(i))
        g.save()
        for i in range(a.rounds):
            t=time.perf_counter(); g.add_node(node(a.initial+i)); g.save(); times["write"].append((time.perf_counter()-t)*1000)
            t=time.perf_counter(); g.update_node(f"n{i % a.initial}", {"mixedRound": i}); g.save(); times["update"].append((time.perf_counter()-t)*1000)
            t=time.perf_counter(); hot=g.query().where_type("post").where("bucket", 0).order_by("score","desc").limit(25).ids(); times["hotQuery"].append((time.perf_counter()-t)*1000)
            t=time.perf_counter(); persisted=g.query_persisted().where_type("post").where("bucket", 0).order_by("score","desc").limit(25).ids(); times["persistedQuery"].append((time.perf_counter()-t)*1000)
            t=time.perf_counter(); found=g.vectors.query(vec(i % a.initial), 10); times["vectorQuery"].append((time.perf_counter()-t)*1000)
            if not hot or not persisted or len(found)!=10: raise RuntimeError("mixed workload verification failed")
        final_count=g.stats()["persistedNodeCount"]; g.close_store()
    finally: shutil.rmtree(directory, ignore_errors=True)
    result={"schemaVersion":1,"lang":"python","initial":a.initial,"rounds":a.rounds,"dimensions":32,"schedule":"write,update,hotQuery,persistedQuery,vectorQuery","totalMs":(time.perf_counter()-start)*1000,"finalNodeCount":final_count,"operations":{k:summary(v) for k,v in times.items()}}
    a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2)); print(f"Wrote {a.out}")
if __name__ == "__main__": main()

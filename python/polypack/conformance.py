"""Python conformance runner.

Consumes the same language-neutral fixtures as the TypeScript runner
(`tests/conformance/`). Python v1 scope implements every graph fixture except
`hot-cache-eviction` (working-set bound) and `recovery/*` (filesystem
persistence lands in Phase 5).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from . import (
    ExactIndex,
    GraphQuery,
    HnswIndex,
    PolyGraph,
    PolypackClosedError,
    PolypackDimensionError,
    PolypackError,
    PolypackValueError,
)

_FIXTURES = Path(__file__).resolve().parent.parent.parent / "fixtures" / "conformance"


class ConformanceError(AssertionError):
    pass


def _map_error_code(err: Exception) -> str:
    if isinstance(err, PolypackDimensionError):
        return "dimension_mismatch"
    if isinstance(err, (PolypackValueError,)):
        message = str(err)
        if "dimension" in message.lower():
            return "dimension_mismatch"
        return "invalid_argument" if "must not be empty" in message or "must contain" in message else "range_out_of_bounds"
    if isinstance(err, PolypackClosedError):
        return "closed"
    return "storage"


def _assert_error(err: Optional[Exception], expected: Optional[str]) -> None:
    if expected is None:
        if err is not None:
            raise ConformanceError(f"operation threw unexpectedly: {err}")
        return
    if err is None:
        raise ConformanceError(f"expected error {expected} but operation succeeded")
    got = _map_error_code(err)
    if got != expected:
        raise ConformanceError(f"expected error {expected}, got {got} ({err})")


def _to_graph_node(node: dict) -> dict:
    out = dict(node)
    if node.get("vector") is not None:
        out["vector"] = list(node["vector"])
    return out


class _Recorder:
    def __init__(self) -> None:
        self.events: list[str] = []

    def __call__(self, id_: str) -> None:
        self.events.append(id_)


def _apply_plan(query: GraphQuery, plan: dict) -> None:
    if plan.get("nodeTypes"):
        query.where_type(*plan["nodeTypes"])
    for attr in plan.get("attributes", []):
        if attr["operator"] == "eq":
            query.where(attr["field"], attr["value"])
        else:
            query.where_range(attr["field"], attr.get("above"), attr.get("below"))
    if plan.get("traversal"):
        for step in plan["traversal"]:
            query.traverse(step["edgeType"], step["depth"], step["direction"])
    if plan.get("joins"):
        for j in plan["joins"]:
            query.join(j["edgeType"], j["direction"])
    if plan.get("similarity"):
        sim = plan["similarity"]
        query.similar_to(sim["vector"], sim["threshold"], sim.get("topK"))
    if plan.get("order"):
        query.order_by(plan["order"]["field"], plan["order"]["direction"])
    if plan.get("offset") is not None:
        query.offset(plan["offset"])
    if plan.get("limit") is not None:
        query.limit(plan["limit"])


def load_fixtures() -> list[dict]:
    return [
        json.loads(p.read_text())
        for p in sorted(_FIXTURES.glob("*.json"))
    ]


def run_fixture(fixture: dict, index: Optional[Any] = None) -> None:
    name = fixture["name"]
    if fixture.get("group") == "hot-cache-eviction":
        raise ConformanceError(f"{name}: hot-cache-eviction is out of Python v1 scope")

    orphan_recorder = _Recorder() if fixture.get("orphanAware") else None
    graph = PolyGraph(on_orphan=orphan_recorder if orphan_recorder else None)
    if index is not None:
        graph.vectors = index

    needs_hnsw = any(op.get("op", "").startswith("hnsw") for op in fixture.get("operations", [])) or bool(
        fixture.get("expect", {}).get("hnswSearches")
    )
    hnsw = HnswIndex(ef_search=300) if needs_hnsw else None

    if hnsw is not None:
        for node in fixture.get("setup", {}).get("nodes", []):
            if node.get("vector"):
                hnsw.add(node["id"], node["vector"])

    for node in fixture.get("setup", {}).get("nodes", []):
        graph.add_node(_to_graph_node(node))
    for edge in fixture.get("setup", {}).get("edges", []):
        graph.add_edge(edge["source"], edge["type"], edge["target"], edge.get("data"), edge.get("ownership"))

    for op in fixture.get("operations", []):
        kind = op["op"]
        err = None
        try:
            if kind == "addNode":
                graph.add_node(_to_graph_node(op["node"]))
            elif kind == "updateNode":
                graph.update_node(op["id"], op.get("data") or {}, op.get("vector"))
            elif kind == "addEdge":
                graph.add_edge(op["source"], op["type"], op["target"], op.get("data"), op.get("ownership"))
            elif kind == "removeNode":
                graph.remove_node(op["id"])
            elif kind == "removeEdges":
                graph.remove_edges(op["source"], op.get("type"), op.get("target"))
            elif kind == "hnswAdd":
                hnsw.add(op["id"], op["vector"])
            elif kind == "hnswRemove":
                hnsw.remove(op["id"])
            elif kind == "hnswUpdate":
                hnsw.update(op["id"], op["vector"])
            elif kind == "mutateDetached":
                node = graph.get_node(op["id"])
                if node is None:
                    raise ConformanceError(f"mutateDetached: node {op['id']} missing")
                node["data"] = dict(node.get("data") or {}, __tampered=True)
                if node.get("vector") is not None:
                    node["vector"][0] = 999
        except (PolypackError, ConformanceError) as e:
            err = e
        _assert_error(err, op.get("expectError"))

    _assert_expectations(graph, orphan_recorder, hnsw, fixture.get("expect", {}))


def _assert_expectations(graph: PolyGraph, recorder: Optional[_Recorder], hnsw: Optional[HnswIndex], expect: dict) -> None:
    for id_ in expect.get("presentNodeIds", []):
        if graph.get_node(id_) is None:
            raise ConformanceError(f"expected node {id_} to exist")
    for id_ in expect.get("absentNodeIds", []):
        if graph.get_node(id_) is not None:
            raise ConformanceError(f"expected node {id_} to be absent")
    if expect.get("nodeCount") is not None:
        if graph.size != expect["nodeCount"]:
            raise ConformanceError(f"nodeCount {graph.size} != {expect['nodeCount']}")
    for id_, fields in (expect.get("nodeData") or {}).items():
        node = graph.get_node(id_)
        if node is None:
            raise ConformanceError(f"nodeData: node {id_} missing")
        data = node.get("data") or {}
        for k, v in fields.items():
            if data.get(k) != v:
                raise ConformanceError(f"nodeData[{id_}].{k} = {data.get(k)} != {v}")
    for id_, vector in (expect.get("nodeVector") or {}).items():
        node = graph.get_node(id_)
        if node is None or node.get("vector") is None:
            raise ConformanceError(f"nodeVector: node {id_} has no vector")
        if node["vector"] != list(vector):
            raise ConformanceError(f"nodeVector[{id_}] mismatch")
    for spec in expect.get("edgeTargets", []):
        got = sorted(graph.get_edge_targets(spec["source"], spec["type"]))
        want = sorted(spec["targets"])
        if got != want:
            raise ConformanceError(f"edgeTargets {spec['source']}/{spec['type']}: {got} != {want}")
    if expect.get("orphanEvents"):
        if recorder is None or recorder.events != expect["orphanEvents"]:
            got = recorder.events if recorder else []
            raise ConformanceError(f"orphanEvents {got} != {expect['orphanEvents']}")
    for spec in expect.get("queries", []):
        query = graph.query()
        _apply_plan(query, spec["plan"])
        got = query.ids()
        if got != spec["resultIds"]:
            raise ConformanceError(f"query result {got} != {spec['resultIds']}")
    for spec in expect.get("exactSearches", []):
        err = None
        got = []
        try:
            got = [r[0] for r in graph.vectors.query(spec["vector"], spec["topK"], spec.get("threshold", 0.0))]
        except PolypackError as e:
            err = e
        if spec.get("expectError"):
            _assert_error(err, spec["expectError"])
        else:
            _assert_error(err, None)
            if got != spec["resultIds"]:
                raise ConformanceError(f"exactSearch {got} != {spec['resultIds']}")
    for spec in expect.get("hnswSearches", []):
        if hnsw is None:
            raise ConformanceError("hnswSearch requires hnsw in the fixture")
        err = None
        got = []
        try:
            got = [r[0] for r in hnsw.query(spec["vector"], spec["topK"], 0.0)]
        except PolypackError as e:
            err = e
        if spec.get("expectError"):
            _assert_error(err, spec["expectError"])
            continue
        _assert_error(err, None)
        overlap = len([i for i in got if i in spec["resultIds"]])
        minimum = spec.get("minOverlap", spec["topK"])
        if overlap < minimum:
            raise ConformanceError(f"hnswSearch overlap {overlap} < {minimum} (got {got}, expected {spec['resultIds']})")
    for id_, vector in (expect.get("hnswVector") or {}).items():
        if hnsw is None:
            raise ConformanceError("hnswVector requires hnsw in the fixture")
        got = hnsw.get(id_)
        if got != list(vector):
            raise ConformanceError(f"hnswVector[{id_}] mismatch")
    if expect.get("hnswSize") is not None:
        if hnsw is None:
            raise ConformanceError("hnswSize requires hnsw in the fixture")
        if len(hnsw) != expect["hnswSize"]:
            raise ConformanceError(f"hnswSize {len(hnsw)} != {expect['hnswSize']}")
    if expect.get("aggregate"):
        spec = expect["aggregate"]
        query = graph.query()
        if spec.get("plan"):
            _apply_plan(query, spec["plan"])
        result = query.aggregate(spec["field"], spec["op"])
        if result != {"value": spec["value"], "count": spec["count"]}:
            raise ConformanceError(f"aggregate {spec['op']}({spec['field']}) = {result} != {spec['value']}/{spec['count']}")


def run_all() -> list[tuple[str, str, Optional[str]]]:
    """Run every fixture, returning (name, status, error_or_None).

    status is 'pass', 'skip', or 'fail'.
    """
    out = []
    for fixture in load_fixtures():
        name = fixture["name"]
        if fixture.get("group") == "hot-cache-eviction":
            out.append((name, "skip", "hot-cache-eviction is out of Python v1 scope"))
            continue
        try:
            run_fixture(fixture)
            out.append((name, "pass", None))
        except ConformanceError as e:
            out.append((name, "fail", str(e)))
    return out

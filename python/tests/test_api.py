"""Smoke tests for the Python API surface."""

import numpy as np
import pytest

import polypack
from polypack import (
    ChangeBatch,
    ExactIndex,
    HnswIndex,
    PolyGraph,
    PolypackDimensionError,
    PolypackError,
    PolypackValueError,
    engine_info,
)


def test_engine_info():
    info = engine_info()
    assert info["vector"] == "rust-native"


def test_exact_index_numpy_and_lists():
    idx = ExactIndex("cosine")
    idx.add("a", np.array([1.0, 0.0, 0.0]))
    idx.add("b", [0.8, 0.6, 0.0])
    assert idx.query(np.array([1.0, 0.0, 0.0]), 2) == [("a", 1.0), ("b", 0.8)]
    assert idx.get("a") == [1.0, 0.0, 0.0]
    assert idx.has("a")
    assert len(idx) == 2
    assert sorted(i for i, _ in idx.entries()) == ["a", "b"]


def test_exact_index_dimension_error():
    idx = ExactIndex()
    idx.add("a", [1.0, 0.0])
    with pytest.raises(PolypackDimensionError):
        idx.query([1.0, 0.0, 0.0], 1)


def test_hnsw_index_churn():
    idx = HnswIndex(ef_search=300)
    idx.add("a", [1.0, 0.0, 0.0])
    idx.add("b", [0.0, 1.0, 0.0])
    idx.remove("a")
    assert len(idx) == 1
    assert not idx.has("a")
    idx.add("a", [1.0, 0.0, 0.0])
    idx.update("b", [1.0, 0.0, 0.0])
    assert sorted(i for i, _ in idx.query([1.0, 0.0, 0.0], 2)) == ["a", "b"]


def test_graph_ownership_cascade():
    graph = PolyGraph()
    graph.add_node({"id": "a", "type": "user", "data": {}, "insertedAt": 1, "updatedAt": 1})
    graph.add_node({"id": "d", "type": "doc", "data": {}, "insertedAt": 1, "updatedAt": 1})
    graph.add_edge("a", "OWNS", "d", ownership="owned")
    graph.remove_node("a")
    assert graph.get_node("d") is None


def test_graph_query_and_vectors():
    graph = PolyGraph()
    for i, v in enumerate([[1, 0, 0], [0.8, 0.6, 0], [0, 1, 0]]):
        graph.add_node(
            {"id": f"n{i}", "type": "doc", "data": {"score": i}, "vector": v, "insertedAt": i, "updatedAt": i}
        )
    assert graph.query().order_by("score", "desc").ids() == ["n2", "n1", "n0"]
    assert graph.query().similar_to([1, 0, 0], 0.5, 1).ids() == ["n0"]
    assert [r[0] for r in graph.vectors.query([1, 0, 0], 2)] == ["n0", "n1"]


def test_validation_errors():
    graph = PolyGraph()
    with pytest.raises(PolypackValueError):
        graph.add_node({"id": "", "type": "t", "data": {}, "insertedAt": 1, "updatedAt": 1})
    with pytest.raises(PolypackError):
        graph.add_edge("a::b", "REL", "c")


def test_context_manager_and_change_batch():
    with PolyGraph() as graph:
        graph.add_node({"id": "a", "type": "t", "data": {}, "insertedAt": 1, "updatedAt": 1})
    assert graph.size == 0
    batch = ChangeBatch(put_nodes=[{"id": "x", "type": "t"}])
    batch.validate()
    with pytest.raises(PolypackValueError):
        ChangeBatch(put_nodes=[{"id": "", "type": "t"}]).validate()

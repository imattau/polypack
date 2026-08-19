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
    PolypackStorageError,
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


@pytest.mark.parametrize("kwarg", ["m", "mmax0", "ef_construction", "ef_search"])
def test_hnsw_index_rejects_non_positive_config(kwarg):
    with pytest.raises(PolypackValueError):
        HnswIndex(**{kwarg: 0})


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


def test_secondary_indexes_filter_equality_and_numeric_ranges():
    graph = PolyGraph()
    graph.define_index("email", ["email"], unique=True)
    graph.define_index("birth-year", ["birthYear"], sparse=True)
    graph.add_node({"id": "a", "type": "person", "data": {"email": "a@test", "birthYear": 1980}, "insertedAt": 1, "updatedAt": 1})
    graph.add_node({"id": "b", "type": "person", "data": {"email": "b@test", "birthYear": 2020}, "insertedAt": 1, "updatedAt": 1})
    graph.add_node({"id": "c", "type": "person", "data": {"email": "c@test", "birthYear": 2050}, "insertedAt": 1, "updatedAt": 1})
    assert graph.query().where("email", "b@test").ids() == ["b"]
    assert sorted(graph.query().where_range("birthYear", above=2000).ids()) == ["b", "c"]
    graph.update_node("b", {"email": "updated@test"})
    assert graph.query().where("email", "b@test").ids() == []
    assert graph.query().where("email", "updated@test").ids() == ["b"]


def test_indexed_query_metrics_report_candidate_scan_count():
    graph = PolyGraph()
    graph.define_index("email", ["email"])
    for id_, email in [("a", "a@test"), ("b", "b@test")]:
        graph.add_node({"id": id_, "type": "person", "data": {"email": email}, "insertedAt": 1, "updatedAt": 1})
    assert graph.query().where("email", "a@test").ids() == ["a"]
    stats = graph.stats()
    assert stats["queryIndexUsage"]["email"] == 1
    assert stats["queryScannedRecords"] == 1


def test_index_catalog_rolls_back_when_metadata_write_fails(tmp_path, monkeypatch):
    graph = PolyGraph.open(str(tmp_path))
    graph.define_index("email", ["email"])

    def fail_persist():
        raise PolypackStorageError("metadata write failed")

    monkeypatch.setattr(graph, "_persist_index_metadata", fail_persist)
    with pytest.raises(PolypackStorageError):
        graph.define_index("birth-year", ["birthYear"])
    assert [index["name"] for index in graph.indexes] == ["email"]
    with pytest.raises(PolypackStorageError):
        graph.drop_index("email")
    assert [index["name"] for index in graph.indexes] == ["email"]


def test_clear_preserves_attached_store_index_constraints(tmp_path):
    graph = PolyGraph.open(str(tmp_path))
    graph.define_index("email", ["email"], unique=True)
    graph.add_node({"id": "a", "type": "person", "data": {"email": "a@test"}, "insertedAt": 1, "updatedAt": 1})
    graph.save()
    graph.clear()
    with pytest.raises(polypack.UniqueConstraintError):
        graph.add_node({"id": "b", "type": "person", "data": {"email": "a@test"}, "insertedAt": 1, "updatedAt": 1})


def test_validation_errors():
    graph = PolyGraph()
    with pytest.raises(PolypackValueError):
        graph.add_node({"id": "", "type": "t", "data": {}, "insertedAt": 1, "updatedAt": 1})
    with pytest.raises(PolypackError):
        graph.add_edge("a::b", "REL", "c")


def test_adapter_capability_requirements():
    graph = PolyGraph()
    graph.require_adapter_capabilities({"transactions": True, "vectorSearch": "exact"})
    with pytest.raises(PolypackValueError, match="fsync"):
        graph.require_adapter_capabilities(fsync=True)


def test_index_metadata_rejects_duplicate_or_empty_definitions(tmp_path):
    (tmp_path / "indexes.json").write_text(
        '[{"name":"dup","fields":["email"]},{"name":"dup","fields":[]}]',
        encoding="utf-8",
    )
    with pytest.raises(PolypackStorageError, match="invalid index metadata"):
        PolyGraph.open(str(tmp_path))


def test_context_manager_and_change_batch():
    with PolyGraph() as graph:
        graph.add_node({"id": "a", "type": "t", "data": {}, "insertedAt": 1, "updatedAt": 1})
    assert graph.size == 0
    batch = ChangeBatch(put_nodes=[{"id": "x", "type": "t"}])
    batch.validate()
    with pytest.raises(PolypackValueError):
        ChangeBatch(put_nodes=[{"id": "", "type": "t"}]).validate()


def _seed_graph(graph):
    graph.add_node({"id": "n1", "type": "doc", "data": {"title": "Hello"}, "vector": [0.1, 0.2, 0.3], "insertedAt": 1, "updatedAt": 1})
    graph.add_node({"id": "n2", "type": "doc", "data": {}, "vector": None, "insertedAt": 2, "updatedAt": 2})
    graph.add_edge("n1", "LINKS", "n2", ownership="reference")


def test_persist_round_trip(tmp_path):
    g = PolyGraph()
    _seed_graph(g)
    g.open_store(str(tmp_path))
    g.save()
    g.close_store()

    g2 = PolyGraph.open(str(tmp_path))
    assert g2.get_node("n1")["vector"] == [0.1, 0.2, 0.3]
    assert g2.get_node("n1")["data"] == {"title": "Hello"}
    assert g2.get_edge_targets("n1", "LINKS") == ["n2"]
    g2.close_store()


def test_latest_mutation_sequence_cursor(tmp_path):
    graph = PolyGraph.open(str(tmp_path))
    graph.add_node({"id": "a", "type": "t", "data": {}, "insertedAt": 1, "updatedAt": 1})
    graph.save()
    assert graph.latest_mutation_sequence() == graph.mutation_log()[-1]["sequence"]


def test_context_manager_saves_before_closing_the_store(tmp_path):
    # No explicit save() — closing the `with` block must not silently lose
    # mutations, mirroring Rust's Graph::close (which flushes before closing).
    with PolyGraph.open(str(tmp_path)) as g:
        _seed_graph(g)

    with PolyGraph.open(str(tmp_path)) as g2:
        assert g2.get_node("n1")["data"] == {"title": "Hello"}
        assert g2.get_edge_targets("n1", "LINKS") == ["n2"]


def test_persist_recovers_from_truncated_wal(tmp_path):
    g = PolyGraph()
    _seed_graph(g)
    g.open_store(str(tmp_path))
    g.save()
    # No snapshot is written until close/compact.
    assert not (tmp_path / "snapshot.msgpack").exists()
    wal = (tmp_path / "wal.msgpack").read_bytes()
    assert len(wal) > 0
    # Crash mid-append: the final in-flight frame is partial. Recovery must
    # still read the acknowledged frames before it.
    (tmp_path / "wal.msgpack").write_bytes(wal[: len(wal) - 2])
    # Release the writer lock without saving or compacting, simulating the
    # original process exiting before recovery starts.
    g._store.close()
    g._store = None

    g2 = PolyGraph.open(str(tmp_path))
    assert g2.get_node("n1") is not None
    assert g2.get_node("n2") is not None
    g2.close_store()
    # Recovery already compacted the truncated WAL down to a snapshot; the
    # save() that close_store() now does first (to avoid discarding unsaved
    # changes) re-applies the unchanged state, so the WAL is empty rather
    # than absent — both mean "nothing left to replay".
    assert (tmp_path / "snapshot.msgpack").exists()
    wal_file = tmp_path / "wal.msgpack"
    assert not wal_file.exists() or wal_file.stat().st_size == 0

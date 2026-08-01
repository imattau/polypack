"""Polypack — embedded local-first property graph with vector search.

Python-native graph layer over the Rust vector core (`polypack._core`).
Semantics mirror `specification/data-model.md` and the TypeScript reference.
"""

from __future__ import annotations

import copy
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Optional, Sequence

import numpy as np

from ._core import (
    ExactIndex as _NativeExactIndex,
    HnswIndex as _NativeHnswIndex,
    NativeStore as _NativeStore,
    engine_info as _engine_info,
    execute_query_plan as _execute_query_plan,
)
from ._core import (
    PolypackClosedError,
    PolypackCorruptDataError,
    PolypackDimensionError,
    PolypackError,
    PolypackStorageError,
    PolypackValueError,
    PolypackVersionError,
)

__all__ = [
    "PolyGraph",
    "GraphQuery",
    "ExactIndex",
    "HnswIndex",
    "ChangeBatch",
    "engine_info",
    "DirectoryStorage",
    "PolypackError",
    "PolypackValueError",
    "PolypackDimensionError",
    "PolypackClosedError",
    "PolypackVersionError",
    "PolypackCorruptDataError",
    "PolypackStorageError",
]

Node = dict
Edge = dict
Ownership = str

_OWNERSHIP_KEY = "__ownership"


def engine_info() -> dict:
    graph, vector, storage = _engine_info()
    return {"graph": graph, "vector": vector, "storage": storage}


# ── Validation helpers ──


def _as_list(vector: Any) -> list:
    if isinstance(vector, np.ndarray):
        return vector.astype(float).tolist()
    return list(vector)


def _validate_vector(vector: Any, name: str = "vector") -> list:
    values = _as_list(vector)
    if not values:
        raise PolypackValueError(f"{name} must not be empty")
    for v in values:
        if not isinstance(v, (int, float)) or not math.isfinite(float(v)):
            raise PolypackValueError(f"{name} must contain finite numbers")
    return [float(v) for v in values]


def _validate_id(id_: str, name: str = "id") -> None:
    if not isinstance(id_, str) or not id_:
        raise PolypackValueError(f"{name} must not be empty")


def _validate_timestamp(ts: Any) -> None:
    if not isinstance(ts, (int, float)) or not math.isfinite(float(ts)) or ts < 0:
        raise PolypackValueError("timestamps must be finite non-negative numbers")


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise PolypackDimensionError(f"expected {len(a)} dimensions, got {len(b)}")
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# ── Vector index wrappers ──


class ExactIndex:
    """Exact vector index backed by the Rust core.

    Mirrors the TypeScript `VectorIndex`: detached reads, validation, and
    `on_change` notifications on mutations.
    """

    def __init__(self, distance: str = "cosine", on_change: Optional[Callable[[str], None]] = None) -> None:
        self._inner = _NativeExactIndex(distance)
        self._on_change = on_change

    def add(self, id_: str, vector: Any) -> None:
        _validate_id(id_, "vector id")
        values = _validate_vector(vector)
        self._inner.add(id_, values)
        if self._on_change:
            self._on_change(id_)

    def add_many(self, entries: Iterable[tuple]) -> None:
        ids = []
        vectors = []
        for id_, vector in entries:
            _validate_id(id_, "vector id")
            ids.append(id_)
            vectors.append(_validate_vector(vector))
        self._inner.add_many(ids, vectors)
        if self._on_change:
            for id_ in ids:
                self._on_change(id_)

    def hydrate(self, id_: str, vector: Any) -> None:
        _validate_id(id_, "vector id")
        self._inner.add(id_, _validate_vector(vector))

    def remove(self, id_: str) -> None:
        self._inner.remove(id_)

    def remove_many(self, ids: Iterable[str]) -> None:
        self._inner.remove_many(list(ids))

    def query(self, vector: Any, top_k: int, threshold: float = 0.0) -> list:
        values = _validate_vector(vector, "query vector")
        if not isinstance(top_k, int) or top_k < 0:
            raise PolypackValueError("topK must be a non-negative integer")
        if not math.isfinite(float(threshold)):
            raise PolypackValueError("threshold must be finite")
        if top_k == 0:
            return []
        return self._inner.query(values, top_k, float(threshold))

    def clear(self) -> None:
        self._inner.clear()

    def has(self, id_: str) -> bool:
        return self._inner.has(id_)

    def get(self, id_: str) -> Optional[list]:
        arr = self._inner.get(id_)
        return None if arr is None else arr.tolist()

    def entries(self) -> Iterator[tuple]:
        for id_, arr in self._inner.entries():
            yield id_, arr.tolist()

    def __len__(self) -> int:
        return len(self._inner)

    @property
    def size(self) -> int:
        return len(self._inner)


class HnswIndex:
    """Approximate vector index backed by the Rust update-safe HNSW."""

    def __init__(
        self,
        on_change: Optional[Callable[[str], None]] = None,
        m: int = 16,
        mmax0: int = 32,
        ef_construction: int = 200,
        ef_search: int = 200,
        level_seed: int = 7,
    ) -> None:
        self._inner = _NativeHnswIndex(m, mmax0, ef_construction, ef_search, level_seed)
        self._on_change = on_change

    def add(self, id_: str, vector: Any) -> None:
        _validate_id(id_, "vector id")
        self._inner.add(id_, _validate_vector(vector))
        if self._on_change:
            self._on_change(id_)

    def update(self, id_: str, vector: Any) -> None:
        _validate_id(id_, "vector id")
        self._inner.update(id_, _validate_vector(vector))
        if self._on_change:
            self._on_change(id_)

    def add_many(self, entries: Iterable[tuple]) -> None:
        ids = []
        vectors = []
        for id_, vector in entries:
            _validate_id(id_, "vector id")
            ids.append(id_)
            vectors.append(_validate_vector(vector))
        self._inner.add_many(ids, vectors)
        if self._on_change:
            for id_ in ids:
                self._on_change(id_)

    def remove(self, id_: str) -> None:
        self._inner.remove(id_)

    def remove_many(self, ids: Iterable[str]) -> None:
        self._inner.remove_many(list(ids))

    def query(self, vector: Any, top_k: int, threshold: float = 0.0) -> list:
        values = _validate_vector(vector, "query vector")
        if not isinstance(top_k, int) or top_k < 0:
            raise PolypackValueError("topK must be a non-negative integer")
        if not math.isfinite(float(threshold)):
            raise PolypackValueError("threshold must be finite")
        if top_k == 0:
            return []
        return self._inner.query(values, top_k, float(threshold))

    def clear(self) -> None:
        self._inner.clear()

    def has(self, id_: str) -> bool:
        return self._inner.has(id_)

    def get(self, id_: str) -> Optional[list]:
        arr = self._inner.get(id_)
        return None if arr is None else arr.tolist()

    def entries(self) -> Iterator[tuple]:
        for id_, arr in self._inner.entries():
            yield id_, arr.tolist()

    def __len__(self) -> int:
        return len(self._inner)

    @property
    def size(self) -> int:
        return len(self._inner)


# ── Graph ──


class DirectoryStorage:
    """Host byte-storage adapter over a directory (snapshot.msgpack / wal.msgpack)."""

    def __init__(self, directory: str) -> None:
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, name: str) -> Path:
        return self._dir / name

    def read(self, name: str) -> Optional[bytes]:
        path = self._path(name)
        return path.read_bytes() if path.exists() else None

    def write(self, name: str, data: bytes) -> None:
        # Write-then-rename: a crash mid-write leaves the previous snapshot
        # intact instead of a torn file, and the tmp file is fsynced first
        # so the rename can never land ahead of its data on disk.
        target = self._path(name)
        tmp_path = self._dir / f"{name}.tmp"
        with open(tmp_path, "wb") as f:
            f.write(bytes(data))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, target)

    def append(self, name: str, data: bytes) -> None:
        with open(self._path(name), "ab") as f:
            f.write(bytes(data))

    def delete(self, name: str) -> None:
        try:
            self._path(name).unlink()
        except FileNotFoundError:
            pass

    def exists(self, name: str) -> bool:
        return self._path(name).exists()


def _copy_node(node: Node) -> Node:
    return {
        "id": node["id"],
        "type": node["type"],
        "data": copy.deepcopy(node.get("data") or {}),
        "vector": None if node.get("vector") is None else list(node.get("vector")),
        "insertedAt": node["insertedAt"],
        "updatedAt": node["updatedAt"],
    }


def _edge_key(source: str, edge_type: str, target: str) -> str:
    if "::" in source or "::" in edge_type:
        raise PolypackValueError('edge source and type must not contain "::"')
    return f"{source}::{edge_type}::{target}"


class PolyGraph:
    """In-memory property graph with vector search and ownership semantics."""

    def __init__(
        self,
        on_orphan: Optional[Callable[[str], None]] = None,
        vector_index: Optional[ExactIndex] = None,
    ) -> None:
        self._nodes: dict[str, Node] = {}
        self._edges: dict[str, dict] = {}
        self._incoming: dict[str, dict] = {}
        self.vectors = vector_index or ExactIndex()
        self._on_orphan = on_orphan
        self._store: Optional[_NativeStore] = None
        self._removed_node_ids: set = set()
        self._removed_edge_ids: set = set()
        self._removed_vector_ids: set = set()

    # ── context manager ──

    def __enter__(self) -> "PolyGraph":
        return self

    def __exit__(self, *exc) -> None:
        if self._store is not None:
            self.close_store()
        else:
            self.clear()

    # ── node CRUD ──

    def add_node(self, node: Node) -> None:
        """Insert or replace a node. Replacement updates the vector index; data and vector are copied on entry."""
        _validate_id(node.get("id"), "node id")
        _validate_id(node.get("type"), "node type")
        _validate_timestamp(node.get("insertedAt"))
        _validate_timestamp(node.get("updatedAt"))
        stored = {
            "id": node["id"],
            "type": node["type"],
            "data": dict(node.get("data") or {}),
            "insertedAt": node["insertedAt"],
            "updatedAt": node["updatedAt"],
        }
        if node.get("vector") is not None:
            stored["vector"] = _validate_vector(node["vector"])
        else:
            stored["vector"] = None
            self.vectors.remove(stored["id"])
        self._nodes[stored["id"]] = stored
        self._removed_node_ids.discard(stored["id"])
        if stored["vector"] is not None:
            self.vectors.add(stored["id"], stored["vector"])
            self._removed_vector_ids.discard(stored["id"])
        else:
            self._removed_vector_ids.add(stored["id"])

    def update_node(self, id_: str, data: dict, vector: Any = None) -> Optional[Node]:
        """Shallow-merge `data` into a loaded node and optionally replace its vector. Returns `None` if the node isn't loaded."""
        node = self._nodes.get(id_)
        if node is None:
            return None
        node["data"].update(data or {})
        if vector is not None:
            node["vector"] = _validate_vector(vector)
            self.vectors.add(id_, node["vector"])
            self._removed_vector_ids.discard(id_)
        node["updatedAt"] = int(time.time() * 1000)
        return _copy_node(node)

    def get_node(self, id_: str) -> Optional[Node]:
        node = self._nodes.get(id_)
        return None if node is None else _copy_node(node)

    def get_nodes(self, ids: Iterable[str]) -> list:
        return [self.get_node(i) for i in ids if self.get_node(i) is not None]

    def remove_node(self, id_: str, _visited: Optional[set] = None) -> None:
        """Remove `id_` and cascade through 'owned' edges. A target of an 'owned' edge is also
        removed unless another 'owned' source keeps it alive. Cyclic owned edges are detected
        so each node is only removed once; `_visited` is an internal recursion argument."""
        visited = _visited if _visited is not None else set()
        if id_ in visited:
            return
        visited.add(id_)
        node = self._nodes.get(id_)
        if node is None:
            return
        for edge in list(self._edges.get(id_, {}).values()):
            if self._ownership(edge) == "owned" and not self._has_other_owned_source(edge["target"], id_):
                self.remove_node(edge["target"], visited)
        self._cleanup_edges(id_)
        del self._nodes[id_]
        self.vectors.remove(id_)
        self._removed_node_ids.add(id_)
        self._removed_vector_ids.add(id_)

    # ── edge CRUD ──

    def add_edge(
        self,
        source: str,
        edge_type: str,
        target: str,
        data: Optional[dict] = None,
        ownership: Optional[Ownership] = None,
        created_at: Optional[int] = None,
    ) -> None:
        """Add one directed edge. A no-op if an edge with the same source/type/target already exists."""
        _validate_id(source, "edge source")
        _validate_id(edge_type, "edge type")
        _validate_id(target, "edge target")
        key = _edge_key(source, edge_type, target)
        if key in self._edges.get(source, {}):
            return
        full = dict(data or {})
        if ownership is not None:
            full[_OWNERSHIP_KEY] = ownership
        self._edges.setdefault(source, {})[key] = {
            "source": source,
            "type": edge_type,
            "target": target,
            "data": full,
            "createdAt": created_at if created_at is not None else 0,
        }
        incoming = self._incoming.setdefault(target, {})
        incoming[source] = incoming.get(source, 0) + 1
        self._removed_edge_ids.discard(key)

    def get_edges(self, source: str, edge_type: Optional[str] = None) -> list:
        edges = self._edges.get(source, {})
        out = []
        for e in edges.values():
            if edge_type is None or e["type"] == edge_type:
                out.append(dict(e, data=dict(e.get("data") or {})))
        return out

    def get_edge_targets(self, source: str, edge_type: str) -> list:
        return [e["target"] for e in self.get_edges(source, edge_type)]

    def get_edge_sources(self, target: str, edge_type: str) -> list:
        sources = []
        for src in self._incoming.get(target, {}):
            for e in self._edges.get(src, {}).values():
                if e["type"] == edge_type and e["target"] == target:
                    sources.append(src)
                    break
        return sources

    def remove_edges(self, source: str, edge_type: Optional[str] = None, target: Optional[str] = None) -> None:
        """Remove edges from `source` matching `edge_type`/`target` (all outgoing edges if both omitted).
        'owned' edges cascade-delete their target unless another source also owns it; 'shared' edges
        invoke `on_orphan` if the target becomes disconnected."""
        edges = self._edges.get(source, {})
        to_remove = [
            (key, e)
            for key, e in edges.items()
            if (edge_type is None or e["type"] == edge_type) and (target is None or e["target"] == target)
        ]
        if not to_remove:
            return
        for _, e in to_remove:
            if self._ownership(e) == "owned" and not self._has_other_owned_source(e["target"], source):
                self.remove_node(e["target"])
        for key, e in to_remove:
            # Cascade deletion may have already removed this edge (and its
            # incoming-index entry) via remove_node above.
            removed = self._edges.get(source, {}).pop(key, None)
            if removed is not None:
                self._decrement_incoming(e["target"], source)
                self._removed_edge_ids.add(key)
            if self._ownership(e) == "shared" and not self._has_incoming(e["target"], source):
                if self._on_orphan:
                    self._on_orphan(e["target"])
        if not self._edges.get(source):
            self._edges.pop(source, None)

    # ── queries ──

    def query(self) -> "GraphQuery":
        """Create a mutable `GraphQuery` over the currently loaded nodes."""
        return GraphQuery(self)

    def clear(self) -> None:
        """Clear in-memory state only — does not flush pending deletions or touch the attached store."""
        self._nodes.clear()
        self._edges.clear()
        self._incoming.clear()
        self.vectors.clear()
        self._removed_node_ids.clear()
        self._removed_edge_ids.clear()
        self._removed_vector_ids.clear()

    # ── persistence (Rust storage state machine) ──

    @classmethod
    def open(cls, directory: str) -> "PolyGraph":
        """Open a directory-backed binary store and load its graph."""
        graph = cls()
        graph.open_store(directory)
        return graph

    def open_store(self, directory: str) -> None:
        """Attach a directory-backed store and load any existing state."""
        self._store = _NativeStore(DirectoryStorage(directory))
        self._load_from_store()

    def close_store(self) -> None:
        """Compact and close the attached store. Safe to call repeatedly."""
        if self._store is not None:
            self._store.close()
            self._store = None

    def save(self) -> None:
        """Persist the full current graph state through the attached store.

        Also flushes deletions recorded since the last successful save, so
        nodes/edges/vectors removed via remove_node/remove_edges do not
        resurrect on the next open().
        """
        if self._store is None:
            raise PolypackStorageError("no store open; call open_store(path) first")
        nodes = []
        for node in self._nodes.values():
            nodes.append(_copy_node(node))
        edges = []
        for edge_map in self._edges.values():
            for e in edge_map.values():
                edges.append(
                    {
                        "id": _edge_key(e["source"], e["type"], e["target"]),
                        "source": e["source"],
                        "target": e["target"],
                        "type": e["type"],
                        "data": dict(e.get("data") or {}),
                        "createdAt": e.get("createdAt", 0),
                    }
                )
        vectors = [{"id": id_, "vector": vector} for id_, vector in self.vectors.entries()]
        self._store.apply(
            put_nodes=nodes,
            delete_node_ids=list(self._removed_node_ids),
            put_edges=edges,
            delete_edge_ids=list(self._removed_edge_ids),
            put_vectors=vectors,
            delete_vector_ids=list(self._removed_vector_ids),
        )
        self._removed_node_ids.clear()
        self._removed_edge_ids.clear()
        self._removed_vector_ids.clear()

    def _load_from_store(self) -> None:
        if self._store is None:
            return
        for id_ in self._store.all_node_ids():
            node = self._store.get_node(id_)
            if node:
                self.add_node(node)
        for edge in self._store.all_edges():
            data = dict(edge.get("data") or {})
            self.add_edge(
                edge["source"],
                edge["type"],
                edge["target"],
                data,
                created_at=edge.get("createdAt"),
            )
        for id_, vector in self._store.all_vectors():
            if id_ not in self._nodes:
                self.vectors.add(id_, vector)

    @property
    def size(self) -> int:
        return len(self._nodes)

    # ── ownership internals ──

    def _ownership(self, edge: dict) -> Ownership:
        return edge.get("data", {}).get(_OWNERSHIP_KEY, "reference")

    def _has_other_owned_source(self, target: str, exclude: str) -> bool:
        for src in self._incoming.get(target, {}):
            if src == exclude:
                continue
            for e in self._edges.get(src, {}).values():
                if e["target"] == target and self._ownership(e) == "owned":
                    return True
        return False

    def _has_incoming(self, target: str, exclude: str) -> bool:
        for src in self._incoming.get(target, {}):
            if src == exclude:
                continue
            if any(e["target"] == target for e in self._edges.get(src, {}).values()):
                return True
        return False

    def _decrement_incoming(self, target: str, source: str) -> None:
        counts = self._incoming.get(target)
        if not counts or source not in counts:
            return
        counts[source] -= 1
        if counts[source] <= 0:
            del counts[source]
        if not counts:
            self._incoming.pop(target, None)

    def _cleanup_edges(self, id_: str) -> None:
        # Remove edges where id_ is the target (from every incoming source).
        for src in list(self._incoming.get(id_, {}).keys()):
            edges = self._edges.get(src, {})
            for key in [k for k, e in edges.items() if e["target"] == id_]:
                del edges[key]
                self._removed_edge_ids.add(key)
            if not edges:
                self._edges.pop(src, None)
        self._incoming.pop(id_, None)
        # Remove edges where id_ is the source, and drop id_ from the
        # incoming index of each of those edges' targets.
        outgoing = self._edges.pop(id_, None)
        if outgoing:
            for key, e in outgoing.items():
                self._decrement_incoming(e["target"], id_)
                self._removed_edge_ids.add(key)


# ── Query builder ──


class GraphQuery:
    def __init__(self, graph: PolyGraph) -> None:
        self._graph = graph
        self._node_types: Optional[list] = None
        self._attributes: list = []
        self._order_by: Optional[tuple] = None
        self._offset: Optional[int] = None
        self._limit: Optional[int] = None
        self._traversal: list = []
        self._joins: list = []
        self._similarity: Optional[dict] = None

    def where_type(self, *types: str) -> "GraphQuery":
        self._node_types = list(types)
        return self

    def where(self, field: str, value: Any) -> "GraphQuery":
        self._attributes.append(("eq", field, value))
        return self

    def where_range(self, field: str, above: Optional[float] = None, below: Optional[float] = None) -> "GraphQuery":
        if above is not None and not math.isfinite(float(above)):
            raise PolypackValueError("range above must be finite")
        if below is not None and not math.isfinite(float(below)):
            raise PolypackValueError("range below must be finite")
        self._attributes.append(("range", field, (above, below)))
        return self

    def order_by(self, field: str, direction: str = "asc") -> "GraphQuery":
        self._order_by = (field, direction)
        return self

    def offset(self, n: int) -> "GraphQuery":
        if not isinstance(n, int) or n < 0:
            raise PolypackValueError("offset must be a non-negative integer")
        self._offset = n
        return self

    def limit(self, n: int) -> "GraphQuery":
        if not isinstance(n, int) or n < 0:
            raise PolypackValueError("limit must be a non-negative integer")
        self._limit = n
        return self

    def traverse(self, edge_type: str, depth: int, direction: str = "out") -> "GraphQuery":
        _validate_id(edge_type, "edge type")
        if not isinstance(depth, int) or depth < 0:
            raise PolypackValueError("depth must be a non-negative integer")
        self._traversal.append((edge_type, depth, direction))
        return self

    def join(self, edge_type: str, direction: str = "out") -> "GraphQuery":
        _validate_id(edge_type, "edge type")
        self._joins.append((edge_type, direction))
        return self

    def similar_to(self, vector: Any, threshold: float = 0.0, top_k: Optional[int] = None) -> "GraphQuery":
        q = _validate_vector(vector, "query vector")
        if not math.isfinite(float(threshold)):
            raise PolypackValueError("threshold must be finite")
        self._similarity = {"vector": q, "threshold": float(threshold), "top_k": top_k}
        return self

    def _match(self, node: Node) -> bool:
        if self._node_types is not None and node["type"] not in self._node_types:
            return False
        data = node.get("data") or {}
        for op, field, value in self._attributes:
            if op == "eq":
                actual = node["type"] if field == "type" else data.get(field)
                if actual != value:
                    return False
            else:
                above, below = value
                actual = data.get(field)
                if actual is None:
                    return False
                if above is not None and actual <= above:
                    return False
                if below is not None and actual >= below:
                    return False
        return True

    def _connected(self, node: Node, edge_type: str, direction: str) -> bool:
        if direction == "out":
            return any(e["type"] == edge_type for e in self._graph.get_edges(node["id"]))
        return any(e["type"] == edge_type for e in self._graph.get_edge_sources(node["id"], edge_type))

    def _bfs(self, seeds: list, edge_type: str, depth: int, direction: str) -> list:
        """Breadth-first expansion; returns nodes in discovery order (seeds first)."""
        visited = list(seeds)
        seen = set(seeds)
        frontier = list(seeds)
        for _ in range(depth):
            if not frontier:
                break
            nxt = []
            for node_id in frontier:
                if direction == "out":
                    targets = [e["target"] for e in self._graph.get_edges(node_id, edge_type)]
                    for t in targets:
                        if t not in seen:
                            seen.add(t)
                            visited.append(t)
                            nxt.append(t)
                else:
                    sources = self._graph.get_edge_sources(node_id, edge_type)
                    for s in sources:
                        if s not in seen:
                            seen.add(s)
                            visited.append(s)
                            nxt.append(s)
            frontier = nxt
        return visited

    def _to_plan(self) -> dict:
        """Express this query as the shared query-plan IR."""
        plan: dict = {}
        if self._node_types:
            plan["nodeTypes"] = list(self._node_types)
        if self._attributes:
            attrs = []
            for op, field, value in self._attributes:
                if op == "eq":
                    attrs.append({"field": field, "operator": "eq", "value": value})
                else:
                    above, below = value
                    entry = {"field": field, "operator": "range"}
                    if above is not None:
                        entry["above"] = above
                    if below is not None:
                        entry["below"] = below
                    attrs.append(entry)
            plan["attributes"] = attrs
        if self._traversal:
            plan["traversal"] = [
                {"edgeType": et, "direction": d, "depth": dep} for et, dep, d in self._traversal
            ]
        if self._joins:
            plan["joins"] = [{"edgeType": et, "direction": d} for et, d in self._joins]
        if self._similarity:
            sim = {
                "vector": self._similarity["vector"],
                "threshold": self._similarity["threshold"],
            }
            if self._similarity["top_k"] is not None:
                sim["topK"] = self._similarity["top_k"]
            plan["similarity"] = sim
        if self._order_by:
            field, direction = self._order_by
            plan["order"] = {"field": field, "direction": direction}
        if self._offset is not None:
            plan["offset"] = self._offset
        if self._limit is not None:
            plan["limit"] = self._limit
        return plan

    def _native_ids(self, plan: dict) -> Optional[list]:
        """Run the plan through the Rust query executor; None on failure."""
        try:
            nodes = [dict(n) for n in self._graph._nodes.values()]
            edges = []
            for edge_map in self._graph._edges.values():
                for e in edge_map.values():
                    edges.append(
                        {
                            "id": _edge_key(e["source"], e["type"], e["target"]),
                            "source": e["source"],
                            "target": e["target"],
                            "type": e["type"],
                            "data": dict(e.get("data") or {}),
                            "createdAt": e.get("createdAt", 0),
                        }
                    )
            return list(_execute_query_plan(nodes, edges, plan))
        except (TypeError, PolypackValueError):
            # Expected fallback: e.g. a non-JSON-serialisable filter value.
            # Other exceptions (native bugs, corrupt state) propagate.
            return None

    def _collect(self) -> list:
        plan = self._to_plan()
        native_ids = self._native_ids(plan)
        if native_ids is not None:
            by_id = {n["id"]: n for n in self._graph._nodes.values()}
            return [by_id[i] for i in native_ids if i in by_id]
        # Fallback: pure-Python pipeline (only reached if the native path
        # failed, e.g. a non-JSON-serialisable filter value).
        return self._collect_python(plan)

    def _collect_python(self, plan: dict) -> list:
        results = [n for n in self._graph._nodes.values() if self._match(n)]
        if self._joins:
            results = [n for n in results if all(self._connected(n, et, d) for et, d in self._joins)]
        for edge_type, depth, direction in self._traversal:
            ids = [n["id"] for n in results]
            order = self._bfs(ids, edge_type, depth, direction)
            by_id = {n["id"]: n for n in self._graph._nodes.values()}
            results = [by_id[i] for i in order if i in by_id]
        if self._order_by:
            field, direction = self._order_by
            results = sorted(
                results,
                key=lambda n: (n.get("data") or {}).get(field, 0) or 0,
                reverse=(direction == "desc"),
            )
        if self._similarity:
            sim = self._similarity
            scored = []
            for n in results:
                if n.get("vector") is None:
                    continue
                score = _cosine(sim["vector"], n["vector"])
                if score >= sim["threshold"]:
                    scored.append((score, n))
            scored.sort(key=lambda x: x[0], reverse=True)
            if sim["top_k"] is not None:
                scored = scored[: sim["top_k"]]
            results = [n for _, n in scored]
        if self._offset is not None:
            results = results[self._offset :]
        if self._limit is not None:
            results = results[: self._limit]
        return results

    def to_list(self) -> list:
        return [_copy_node(n) for n in self._collect()]

    def ids(self) -> list:
        return [n["id"] for n in self._collect()]

    def first(self) -> Optional[Node]:
        nodes = self.to_list()
        return nodes[0] if nodes else None

    def count(self) -> int:
        return len(self._collect())

    def aggregate(self, field: str, op: str) -> dict:
        values = [
            (n.get("data") or {}).get(field)
            for n in self._collect()
            if isinstance((n.get("data") or {}).get(field), (int, float))
            and math.isfinite(float((n.get("data") or {}).get(field)))
        ]
        if not values:
            return {"value": 0, "count": 0}
        if op == "sum":
            value = sum(values)
        elif op == "avg":
            value = sum(values) / len(values)
        elif op == "min":
            value = min(values)
        elif op == "max":
            value = max(values)
        elif op == "count":
            value = len(values)
        else:
            raise PolypackValueError(f"unknown aggregate op {op}")
        return {"value": value, "count": len(values)}


# ── Change batch (v1: structural contract; execution is Phase 5) ──


@dataclass
class ChangeBatch:
    put_nodes: list = field(default_factory=list)
    delete_node_ids: list = field(default_factory=list)
    put_edges: list = field(default_factory=list)
    delete_edge_ids: list = field(default_factory=list)
    put_vectors: list = field(default_factory=list)
    delete_vector_ids: list = field(default_factory=list)

    def validate(self) -> None:
        for n in self.put_nodes:
            _validate_id(n.get("id"), "node id")
            _validate_id(n.get("type"), "node type")
        for e in self.put_edges:
            _validate_id(e.get("source"), "edge source")
            _validate_id(e.get("type"), "edge type")
            _validate_id(e.get("target"), "edge target")

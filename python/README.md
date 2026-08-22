# polypack (Python)

Python bindings for the polypack embedded property-graph and vector-search
core. `PolyGraph` and `GraphQuery` are a Python layer mirroring the
[TypeScript `PolyGraph`/`GraphQuery` API](../docs/API.md); vector indexing,
persisted queries, and directory-backed persistence run in the Rust core
(`polypack._core`, built from [`crates/polypack-python`](../crates/polypack-python)).
Simple hot-cache filters, ordering, pagination, and aggregates use the local
Python pipeline to avoid repeatedly serializing the working set across the
Python/Rust boundary.

## Install

```sh
pip install polypack-db
```

Requires Python 3.8+ and `numpy`. Prebuilt wheels are published from
[`polypack-db` on PyPI](https://pypi.org/project/polypack-db/); the version
tracks the TypeScript package and `polypack-core` crate in lockstep.

## Quick start

```python
import time
from polypack import PolyGraph

graph = PolyGraph()

graph.add_node({
    "id": "doc_1",
    "type": "document",
    "data": {"title": "Quantum Computing"},
    "vector": [0.95, 0.20, 0.10],
    "insertedAt": int(time.time() * 1000),
    "updatedAt": int(time.time() * 1000),
})

graph.add_edge("doc_1", "REFERENCES", "doc_2", ownership="owned")

# Search by similarity
graph.query() \
    .where_type("document") \
    .similar_to([0.90, 0.30, 0.10], 0.5, 5) \
    .to_list()

# Filter, traverse, aggregate
graph.query().where("title", "Quantum Computing").traverse("REFERENCES", 3).to_list()
graph.query().where_type("book").aggregate("price", "avg")
```

For directory-backed stores, use `graph.query_persisted()` for filters,
ordering, pagination, and counts. These operations execute in the native store
and only materialize the requested result page in Python:

```python
graph.query_persisted().where_type("book").order_by("price", "desc").limit(20).to_list()
```

`query_persisted()` requires an open directory-backed store and returns a
`PersistedGraphQuery`. Its supported chainable methods are:

- `where_type(*types)`
- `where(field, value)`
- `where_range(field, above=None, below=None)` with exclusive boundaries
- `order_by(field, direction="asc")`
- `offset(n)` and `limit(n)`

Its terminal methods are synchronous: `to_list()`, `ids()`, and `count()`.
`to_list()` returns only the requested page, while `count()` counts all records
matching the filters before pagination. Pending writable changes are saved
before a persisted query runs. Similarity, joins, and traversal remain on the
hot `GraphQuery` API.

### Python API surface

The main public classes are:

- `PolyGraph`: graph lifecycle, mutations, transactions, persistence, and hot queries.
- `GraphQuery`: synchronous hot-cache filters, ranges, ordering, pagination, traversal, joins, similarity, activation, and aggregates.
- `PersistedGraphQuery`: native storage-level filtering, ordering, pagination, and counting for directory-backed stores.
- `ExactIndex` and `HnswIndex`: standalone native vector indexes.
- `ActivationEngine`: adaptive-memory operations over a `PolyGraph`.

Python method names use `snake_case` equivalents of the TypeScript API, such as
`add_node`, `update_node`, `where_type`, `similar_to`, and `to_list`.

## Persistence

`PolyGraph.open(directory)` attaches a directory-backed binary store
(snapshot + write-ahead log, shared format with the TypeScript
`BinaryStoreAdapter`) and loads any existing state. Use it as a context
manager so the store is compacted and closed on exit:

```python
with PolyGraph.open("./data") as graph:
    graph.add_node({...})
    graph.save()  # persists queued puts and deletions
```

## Activation (adaptive memory)

Nodes can carry durable activation state that persists through the store.
Reinforcement decay-corrects the prior state, adds a delta, and re-anchors the
decay clock; reads decay lazily as a pure function of elapsed time:

```python
from polypack import PolyGraph, ActivationEngine

graph = PolyGraph()
graph.add_node({"id": "a", "type": "doc", "data": {}, "insertedAt": 1, "updatedAt": 1})
graph.reinforce_node("a", 0.6, "user_read")      # durable, persisted
graph.reinforce_node("a", 0.3, "user_read", context="project-x")  # also reinforces a context
graph.suppress_node("outdated", 1.0, "stale")    # durable inhibition
graph.get_activation("a")                         # ~0.6, decay-corrected
graph.top_activated(5)                            # working-memory primitive

engine = ActivationEngine(graph)
engine.bump_attention("a", 0.2)                  # local, never serialized
engine.spread(["a"], depth=2, decay=0.5)         # relational spreading activation
engine.pulse([1.0, 0.0, ...])                    # semantic region scoring (vector)
engine.absorb([1.0, 0.0, ...])                   # pulse + reinforce above threshold
engine.working_memory(5)                         # current "mental state"
engine.working_memory(limit=8, token_budget=2000, diversity_lambda=0.5)  # budgeted + diverse
engine.working_memory(context="project-x", context_fallback=True)  # context lens with global fallback
engine.working_memory_persisted(limit=8, token_budget=2000)  # rank cold persisted nodes
engine.score_breakdown_of(graph.get_node("a"), 0.8, 0.2)  # explain pulse score
engine.set_weights({"semantic": 1.2})  # restore application-persisted feedback weights
engine.record_feedback("a", was_useful=True)     # nudge pulse's scoring weights
```

`graph.query().where_activated(0.4)` and `graph.query().order_by_activation()`
filter/order by current activation. `merge_activation` and `decay_factor` are
exposed for parity with the TypeScript/Rust implementations.

Nodes also support memory classes (`episodic`/`semantic`/`procedural`/`entity`,
set via `register_node_type(type, memory_class=...)` as a type default or
`memoryClass` on the node itself, driving differentiated decay half-lives) and
confidence/provenance fields (`confidence`, `source`, `observedAt`,
`derivedFrom`, `supersedes`, `contradicts`) — ordinary node fields, not
activation state. `graph.supersede(id, superseded_id)` records a contradiction
(suppresses the superseded node without deleting it) and
`graph.consolidate(node, source_ids)` promotes a group of nodes into a
higher-level one, both while keeping the sources in the graph. See
[docs/API.md](../docs/API.md) for full parameter details.

`save()` writes the complete current graph and flushes any deletions
recorded since the last save. `close_store()` — and exiting the `with`
block, which calls it — saves any unsaved changes first, then compacts
and closes; you don't need to call `save()` yourself before it, though
doing so periodically (as above) still bounds how much a single close
has to write and how much a crash before that could lose.

## Vector indexes

`ExactIndex` and `HnswIndex` wrap the Rust vector core directly and can be
used standalone, independent of `PolyGraph`:

```python
from polypack import ExactIndex, HnswIndex

index = ExactIndex(distance="cosine")
index.add("doc_1", [0.95, 0.20, 0.10])
index.query([0.90, 0.30, 0.10], top_k=5, threshold=0.5)

approx = HnswIndex(m=16, ef_search=200)
approx.add_many([("doc_1", [0.95, 0.20, 0.10]), ("doc_2", [0.10, 0.90, 0.05])])
```

## Errors

Native errors surface as `PolypackError` subclasses: `PolypackValueError`,
`PolypackDimensionError`, `PolypackClosedError`, `PolypackVersionError`,
`PolypackCorruptDataError`, and `PolypackStorageError`.

## Development

See the top-level [CONTRIBUTING.md](../CONTRIBUTING.md) for building the
native extension with maturin and running the Python test suite.

# polypack (Python)

Python bindings for the polypack embedded property-graph and vector-search
core. `PolyGraph` and `GraphQuery` are a native Python layer mirroring the
[TypeScript `PolyGraph`/`GraphQuery` API](../docs/API.md); vector indexing,
query execution, and directory-backed persistence run in the Rust core
(`polypack._core`, built from [`crates/polypack-python`](../crates/polypack-python)).

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
graph.get_activation("a")                         # ~0.6, decay-corrected
graph.top_activated(5)                            # working-memory primitive

engine = ActivationEngine(graph)
engine.bump_attention("a", 0.2)                  # local, never serialized
engine.spread(["a"], depth=2, decay=0.5)         # relational spreading activation
engine.pulse([1.0, 0.0, ...])                    # semantic region scoring (vector)
engine.absorb([1.0, 0.0, ...])                   # pulse + reinforce above threshold
engine.working_memory(5)                         # current "mental state"
```

`graph.query().where_activated(0.4)` and `graph.query().order_by_activation()`
filter/order by current activation. `merge_activation` and `decay_factor` are
exposed for parity with the TypeScript/Rust implementations.

`save()` writes the complete current graph and flushes any deletions
recorded since the last save; `close_store()` compacts and closes explicitly.

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

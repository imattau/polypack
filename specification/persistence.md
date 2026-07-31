# Persistence specification

Version: 1 (draft)

## 1. Model

Persistence stores three record sets — nodes, edges, and vectors — as:

1. A **snapshot**: a single MessagePack document describing the complete state.
2. A **write-ahead log (WAL)**: an append-only sequence of framed entries
   describing every change since the snapshot was written.

The WAL is replayed on top of the snapshot at startup.

## 2. File layout

A store directory contains two files:

- `snapshot.msgpack` — MessagePack snapshot.
- `wal.msgpack` — framed WAL entries.

Files are treated as opaque byte streams; the storage adapter (filesystem,
OPFS, memory) owns bytes only.

## 3. Snapshot format (version 1)

A single MessagePack map:

```text
{
  "version": 1,
  "nodes":   [["id", node], ...],   // node = change-batch node envelope
  "edges":   [["id", edge], ...],   // edge = change-batch edge envelope
  "vectors": [["id", [number, ...]], ...]
}
```

`version` is an integer. Readers must reject snapshots with an unsupported
version with a precise version error.

## 4. WAL format (version 1)

A WAL is a byte sequence of frames. Each frame is:

```text
4-byte big-endian unsigned length N
N bytes of MessagePack-encoded entry
```

An entry is one of:

```text
{ "kind": "putNode",    "node": node }
{ "kind": "deleteNode", "id": id }
{ "kind": "putEdge",    "edge": edge }
{ "kind": "deleteEdge", "id": id }
{ "kind": "putVector",  "id": id, "vector": [number, ...] }
{ "kind": "deleteVector", "id": id }
{ "kind": "clearAll" }
```

Decoding stops at the first truncated or invalid frame; the trailing partial
frame is ignored. A crash mid-append therefore loses at most the in-flight
frame, never the acknowledged frames before it.

## 5. Recovery order

Startup:

1. Read snapshot, if present.
2. Read WAL; if non-empty, replay every complete frame.
3. Write a new snapshot capturing snapshot + replayed WAL state.
4. Only then delete the WAL.

A crash between steps 3 and 4 re-replays an already-applied, idempotent WAL, so
no acknowledged change is lost. The old delete-first ordering is prohibited.

## 6. Writes

- Every logical change batch is appended to the WAL as its sequence of frames
  in the batch's deterministic order (deletions first, then insertions, per the
  change-batch contract).
- The in-memory count of WAL entries since the last snapshot is tracked.
- When the count reaches `compactThreshold` (default 10,000), compaction is
  scheduled (debounced).

## 7. Compaction

Compaction runs under the same serial operation queue as writes and startup:

1. Capture the current WAL generation `g` (entries since the snapshot).
2. Encode a snapshot of current in-memory state.
3. Write the snapshot **atomically** (temporary file + rename where the host
   filesystem permits; otherwise an atomic replace).
4. Read the WAL; keep only the entries beyond generation `g` (entries that
   arrived mid-compaction), or truncate to empty.
5. Update the in-memory WAL count to the number of retained entries.

A write that arrives during compaction is reflected in the retained WAL tail
and replayed on the next start, never lost.

## 8. Serialisation

Startup, mutation, compaction, and shutdown are serialised through one
operation queue. Concurrent callers queue; each batch is applied exactly once.
`close()` flushes a pending compaction, then marks the store closed; further
operations reject.

## 9. Durability modes

| Mode      | Guarantee                                                          |
|-----------|--------------------------------------------------------------------|
| `memory`  | Bytes exist in the host process only.                              |
| `process` | Writes reach the host OS (default); survives process crashes, not power loss. |
| `fsync`   | WAL appends and snapshot writes are `fsync`-ed (including the containing directory where supported). |

## 10. Versioning

- Snapshot and WAL formats each carry a version.
- Format changes require an explicit format-version bump.
- Old snapshots either migrate to the current version or fail with a precise
  version error; silent misinterpretation is prohibited.

## 11. Conformance

Recovery fixtures exercise: clean WAL, snapshot only, partial (truncated) WAL
tail, and corrupt mid-stream frame. Acknowledged durable batches must survive
forced termination in every case.

## 12. Implementation status

- The TypeScript reference (`src/persistence/binary-store.ts` +
  `binary-format.ts`) is the current production implementation.
- `crates/polypack-core/src/storage` provides a Rust state machine with
  byte-for-byte compatible v1 codecs. The shared recovery fixtures pass
  against it, and cross-language round-trip tests confirm the TypeScript and
  Rust stores read each other's files. Python (`NativeStore`) and Node native
  (`NativeStore`) bindings expose it; the TypeScript adapter remains pure-TS
  for now.

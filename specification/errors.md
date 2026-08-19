# Error taxonomy

Version: 1 (draft)

Every Polypack error carries a stable code so all languages can map errors to a
common taxonomy. The TypeScript implementation throws `TypeError`, `RangeError`,
`Error`, or a custom subclass; the mapping below names the code and the 
condition.

## Error codes

| Code                | TypeScript class | Condition                                                            |
|---------------------|------------------|----------------------------------------------------------------------|
| `invalid_argument`  | `TypeError`      | Empty or invalid ID / edge source / edge type / edge target; invalid query parameters; non-finite vector or threshold input. |
| `dimension_mismatch`| `RangeError`     | Query vector and stored vector dimensions differ; index build/query mismatch. |
| `range_out_of_bounds`| `RangeError`    | Negative `topK`, `offset`, `limit`, or traversal `depth`; non-finite range bounds. |
| `conflict`           | `ConflictError` | Conditional write supplied a stale record revision.                    |
| `closed`            | `Error`          | Operation attempted on a closed adapter or store.                    |
| `format_version`    | `Error`          | Snapshot or WAL version unsupported.                                 |
| `corrupt_data`      | `Error`          | Snapshot or WAL bytes fail to decode structurally.                   |
| `storage`           | `Error`          | Host storage I/O failure (disk, OPFS, IndexedDB, etc.).              |
| `not_implemented`   | `Error`          | Required optional hook absent and no fallback path exists.           |
| `resource_limit`    | `ResourceLimitError` | Configured traversal, result, batch, payload, or vector limit exceeded. |

## Mapping rules

- Validation failures always precede mutation; a batch that fails validation
  is rejected as a whole (`all-or-reject`).
- Persistence failures are surfaced as `storage`, never silently swallowed on
  the public mutation path (compaction background errors are logged).
- Dimension mismatches are always `dimension_mismatch`, never coerced.
- ANN engines are not required to throw on approximate differences; recall
  tolerance is a conformance concern, not an error.

## Language mapping

| Code                | Python exception            | Rust type                  |
|---------------------|-----------------------------|----------------------------|
| `invalid_argument`  | `PolypackValueError`        | `PolypackError::InvalidArgument` |
| `dimension_mismatch`| `PolypackDimensionError`    | `PolypackError::DimensionMismatch` |
| `range_out_of_bounds`| `PolypackValueError`       | `PolypackError::InvalidArgument` |
| `conflict`           | `ConflictError`             | `PolypackError::Conflict` |
| `closed`            | `PolypackClosedError`       | `PolypackError::Closed`    |
| `format_version`    | `PolypackVersionError`      | `PolypackError::FormatVersion` |
| `corrupt_data`      | `PolypackCorruptDataError`  | `PolypackError::CorruptData` |
| `storage`           | `PolypackStorageError`      | `PolypackError::Storage`   |
| `not_implemented`   | `NotImplementedError`       | `PolypackError::NotImplemented` |
| `resource_limit`    | `ResourceLimitError`        | `PolypackError::ResourceLimit`  |

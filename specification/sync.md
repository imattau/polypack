# Synchronisation contract

The sync layer is a transport-neutral replication foundation, not an
identity or permission service. Applications may supply authorization and
conflict hooks to the server.

Server operation history has a global cursor. Clients resume with that cursor;
when history has been compacted past the requested cursor, the server returns
`cursor_expired` and a snapshot request starts from the available boundary.
Oversized recovery responses are paginated with `more` and checksummed with a
deterministic operation-batch checksum.

Servers may bound incoming envelopes with `maxBatchOps` and durable work that
has been accepted but not yet written with `maxPendingOps`. Exceeding either
bound rejects the envelope without acknowledging its operations; clients may
retry after reducing the batch or after the durable queue drains. Recovery
pages advance by the global server cursor, even when a subscription filter
removes operations from the page, so filtered clients cannot mistake hidden
operations for an unadvanced cursor.

When authorization or conflict hooks are configured, all operations carrying
one transaction ID are validated as a group. If any member is rejected, no
member is persisted or broadcast. Durable operation logs should implement
`appendBatch` as one logical append so a committed transaction group is not
observed as partially appended by the log contract.

`FileSyncOperationLog` serializes concurrent append and compaction operations
within a process. It is durable through the supplied `FileIO`, but it is not a
multi-process coordination mechanism; deployments requiring multiple writers
must provide external coordination.

Operation IDs are used for idempotent retries. Transaction IDs are carried
through operations so applications can preserve transaction grouping and
perform their own atomic/conflict handling at the receiving boundary.
Durable logs retain operation and transaction identity tombstones across
compaction and expose cursor, retained-operation, and identity counts for
administrative monitoring.

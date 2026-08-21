# Synchronisation contract

The sync layer is a transport-neutral replication foundation, not an
identity or permission service. Applications may supply authorization and
conflict hooks to the server.

The portable protocol surface is shared by TypeScript, Rust, and Python:
operation envelopes use the same required fields and validation rules, and
ordered batches use the same deterministic checksum. The implementations do
not need to share a transport or server runtime. TypeScript provides the
callback-driven transport server; Rust and Python provide synchronous server
state machines suitable for embedding behind any transport.

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

Recovery responses always report a global cursor. If a requested cursor has
expired, the response starts at the retained-history boundary and reports the
cursor immediately after the returned page, including that boundary offset.
Clients can therefore resume pagination without replaying or skipping retained
operations. Subscription filters receive the same per-client metadata as live
broadcasts.

Durable append or compaction failures must not be acknowledged as successful:
the server sends a `persistence_error` acknowledgement and makes its durable
flush operation reject with the underlying error. An append failure leaves the
operations out of the accepted cursor; a compaction failure may follow an
already-committed append, so callers must inspect the cursor before retrying.
Clients may retry after the storage failure is repaired.

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

Native servers expose the same minimum server guarantees: protocol-version
validation, batch limits, global cursors, snapshot/delta recovery, bounded
retention, operation and transaction idempotence, and deterministic checksums.
Authorization, conflict callbacks, subscription predicates, and transport
delivery remain host-language integration points.

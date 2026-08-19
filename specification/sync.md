# Synchronisation contract

The sync layer is a transport-neutral replication foundation, not an
identity or permission service. Applications may supply authorization and
conflict hooks to the server.

Server operation history has a global cursor. Clients resume with that cursor;
when history has been compacted past the requested cursor, the server returns
`cursor_expired` and a snapshot request starts from the available boundary.
Oversized recovery responses are paginated with `more` and checksummed with a
deterministic operation-batch checksum.

`FileSyncOperationLog` serializes concurrent append and compaction operations
within a process. It is durable through the supplied `FileIO`, but it is not a
multi-process coordination mechanism; deployments requiring multiple writers
must provide external coordination.

Operation IDs are used for idempotent retries. Transaction IDs are carried
through operations so applications can preserve transaction grouping and
perform their own atomic/conflict handling at the receiving boundary.

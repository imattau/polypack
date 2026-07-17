/** Ordered graph mutation exchanged by sync clients. */
export interface SyncOp {
  seq: number
  timestamp: number
  clientId: string
  kind: 'addNode' | 'updateNode' | 'removeNode' | 'addEdge' | 'removeEdges'
  payload: Record<string, unknown>
}

/** Transport envelope for deltas, snapshots, acknowledgements, and requests. */
export interface SyncMessage {
  type: 'delta' | 'snapshot' | 'ack' | 'request-snapshot'
  clientId: string
  fromSeq: number
  ops: SyncOp[]
}

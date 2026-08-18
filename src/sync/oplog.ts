import type { SyncOp } from './types.js'

/** Append-only, client-local sequence of graph mutations. */
export class OpLog {
  private ops: SyncOp[] = []
  private nextSeq = 1
  readonly clientId: string

  constructor(clientId: string, existing?: SyncOp[]) {
    this.clientId = clientId
    if (existing) {
      this.ops = [...existing]
      this.nextSeq = existing.length > 0 ? Math.max(...existing.map(o => o.seq)) + 1 : 1
    }
  }

  /** Record a new op, stamping it with the next sequence number, this client's id, and the current time. */
  append(kind: SyncOp['kind'], payload: Record<string, unknown>, options: Pick<SyncOp, 'transactionId' | 'operationId' | 'baseRevision'> = {}): SyncOp {
    const op: SyncOp = {
      seq: this.nextSeq++,
      clientId: this.clientId,
      timestamp: Date.now(),
      kind,
      payload,
      ...options,
    }
    this.ops.push(op)
    return op
  }

  /** All ops since a given sequence number. */
  since(seq: number): SyncOp[] {
    return this.ops.filter(o => o.seq > seq)
  }

  get all(): readonly SyncOp[] {
    return this.ops
  }

  get latestSeq(): number {
    return this.nextSeq - 1
  }

  get size(): number {
    return this.ops.length
  }
}

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
    op.operationId ??= `${this.clientId}:${op.seq}`
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

  /** Drop acknowledged history while retaining all later operations. */
  dropThrough(sequence: number): void {
    const firstPending = this.ops.findIndex(op => op.seq > sequence)
    if (firstPending === -1) this.ops = []
    else if (firstPending > 0) this.ops = this.ops.slice(firstPending)
  }
}

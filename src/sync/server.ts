import type { SyncMessage, SyncOp } from './types.js'

export type SyncServerClient = {
  send: (msg: SyncMessage) => void
  clientId?: string
}

/**
 * Simple relay server. Receives ops from clients, stores them in an
 * in-memory op log, and broadcasts to all other connected clients.
 */
/** In-memory relay that broadcasts received operations to all other clients. */
export class SyncServer {
  private opLog: SyncOp[] = []
  private seenOps = new Set<string>()
  private clients: SyncServerClient[] = []
  onOp?: (op: SyncOp) => void

  /** Register a client transport. Returns a function to handle incoming messages. */
  addClient(handle: SyncServerClient): (msg: SyncMessage) => void {
    this.clients.push(handle)
    return (msg: SyncMessage) => this.handleMessage(msg, handle)
  }

  /** Stop broadcasting to a previously registered client. */
  removeClient(handle: SyncServerClient): boolean {
    const index = this.clients.indexOf(handle)
    if (index === -1) return false
    this.clients.splice(index, 1)
    return true
  }

  private handleMessage(msg: SyncMessage, sender: SyncServerClient): void {
    if (msg.type === 'delta') {
      const accepted: SyncOp[] = []
      for (const op of msg.ops) {
        const key = `${op.clientId}:${op.seq}`
        if (this.seenOps.has(key)) continue
        this.seenOps.add(key)
        this.opLog.push(op)
        accepted.push(op)
        this.onOp?.(op)
      }
      const acknowledgedSeq = msg.ops.reduce((max, op) => Math.max(max, op.seq), msg.fromSeq)
      sender.send({
        type: 'ack',
        clientId: msg.clientId,
        fromSeq: acknowledgedSeq,
        ops: [],
      })
      // Broadcast to all OTHER clients
      for (const client of accepted.length === 0 ? [] : this.clients) {
        if (client === sender) continue
        client.send({
          type: 'delta',
          clientId: 'server',
          fromSeq: 0,
          ops: accepted,
        })
      }
    }
  }

  get ops(): readonly SyncOp[] {
    return this.opLog
  }
}

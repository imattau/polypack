import type { SyncMessage } from './types.js'

/** Minimal bidirectional transport contract consumed by {@link SyncClient}. */
export interface SyncTransport {
  send(msg: SyncMessage): void
  onMessage: ((msg: SyncMessage) => void) | null
  close(): void
}

/** Asynchronous, bidirectional in-memory transport for tests and single-process use. */
export class MemoryTransport implements SyncTransport {
  private peer: MemoryTransport | null = null
  onMessage: ((msg: SyncMessage) => void) | null = null

  /** Link two MemoryTransports so messages flow from one to the other. */
  static pair(): [MemoryTransport, MemoryTransport] {
    const a = new MemoryTransport()
    const b = new MemoryTransport()
    a.peer = b
    b.peer = a
    return [a, b]
  }

  send(msg: SyncMessage): void {
    if (!this.peer) return
    // Deliver asynchronously to simulate network latency
    setTimeout(() => {
      this.peer!.onMessage?.(msg)
    }, 0)
  }

  close(): void {
    this.peer = null
    this.onMessage = null
  }
}

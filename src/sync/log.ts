import type { FileIO } from '../persistence/file-io.js'
import type { SyncOp } from './types.js'
import { syncChecksum } from './checksum.js'

export interface SyncLogState {
  baseCursor: number
  ops: SyncOp[]
  /** Integrity marker for durable files; omitted by legacy logs. */
  checksum?: string
}

export interface SyncOperationLog {
  load(): Promise<SyncLogState>
  append(op: SyncOp): Promise<void>
  compact?(baseCursor: number): Promise<void>
}

/** In-memory log useful for tests and process-local deployments. */
export class MemorySyncOperationLog implements SyncOperationLog {
  private baseCursor = 0
  private ops: SyncOp[] = []

  async load(): Promise<SyncLogState> {
    return { baseCursor: this.baseCursor, ops: this.ops.map(op => ({ ...op, payload: structuredClone(op.payload) })) }
  }

  async append(op: SyncOp): Promise<void> {
    this.ops.push({ ...op, payload: structuredClone(op.payload) })
  }

  async compact(baseCursor: number): Promise<void> {
    const remove = Math.max(0, baseCursor - this.baseCursor)
    this.ops.splice(0, remove)
    this.baseCursor += remove
  }
}

/** Simple durable JSON log for a FileIO-backed sync server. */
export class FileSyncOperationLog implements SyncOperationLog {
  private queue: Promise<unknown> = Promise.resolve()

  constructor(private readonly io: FileIO, private readonly fileName = 'sync-operations.json') {}

  async load(): Promise<SyncLogState> {
    const data = await this.io.readFile(this.fileName)
    if (!data || data.length === 0) return { baseCursor: 0, ops: [] }
    const parsed = JSON.parse(new TextDecoder().decode(data)) as Partial<SyncLogState>
    if (!Number.isInteger(parsed.baseCursor) || (parsed.baseCursor ?? 0) < 0 || !Array.isArray(parsed.ops) || parsed.ops.some(op => !this.validOperation(op))) {
      throw new Error('Invalid sync operation log state')
    }
    if (parsed.checksum !== undefined && parsed.checksum !== syncChecksum(parsed.ops)) {
      throw new Error('Sync operation log checksum mismatch')
    }
    return { baseCursor: parsed.baseCursor as number, ops: parsed.ops, checksum: parsed.checksum }
  }

  private validOperation(op: unknown): op is SyncOp {
    if (!op || typeof op !== 'object') return false
    const candidate = op as Partial<SyncOp>
    return Number.isInteger(candidate.seq) && (candidate.seq ?? -1) >= 0 &&
      typeof candidate.timestamp === 'number' && Number.isFinite(candidate.timestamp) &&
      typeof candidate.clientId === 'string' && candidate.clientId.length > 0 &&
      typeof candidate.kind === 'string' && !!candidate.payload && typeof candidate.payload === 'object'
  }

  async append(op: SyncOp): Promise<void> {
    await this.enqueue(async () => {
      const state = await this.load()
      state.ops.push(op)
      await this.write(state)
    })
  }

  async compact(baseCursor: number): Promise<void> {
    await this.enqueue(async () => {
      const state = await this.load()
      const remove = Math.max(0, baseCursor - state.baseCursor)
      state.ops.splice(0, remove)
      state.baseCursor += remove
      await this.write(state)
    })
  }

  private enqueue<T>(operation: () => Promise<T>): Promise<T> {
    const run = this.queue.then(operation)
    this.queue = run.then(() => undefined, () => undefined)
    return run
  }

  private async write(state: SyncLogState): Promise<void> {
    const persisted = { baseCursor: state.baseCursor, ops: state.ops, checksum: syncChecksum(state.ops) }
    await this.io.writeFile(this.fileName, new TextEncoder().encode(JSON.stringify(persisted)))
  }
}

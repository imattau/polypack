import type { FileIO } from '../persistence/file-io.js'
import type { SyncOp } from './types.js'

export interface SyncLogState {
  baseCursor: number
  ops: SyncOp[]
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
  constructor(private readonly io: FileIO, private readonly fileName = 'sync-operations.json') {}

  async load(): Promise<SyncLogState> {
    const data = await this.io.readFile(this.fileName)
    if (!data || data.length === 0) return { baseCursor: 0, ops: [] }
    const parsed = JSON.parse(new TextDecoder().decode(data)) as SyncLogState
    return { baseCursor: parsed.baseCursor ?? 0, ops: parsed.ops ?? [] }
  }

  async append(op: SyncOp): Promise<void> {
    const state = await this.load()
    state.ops.push(op)
    await this.write(state)
  }

  async compact(baseCursor: number): Promise<void> {
    const state = await this.load()
    const remove = Math.max(0, baseCursor - state.baseCursor)
    state.ops.splice(0, remove)
    state.baseCursor += remove
    await this.write(state)
  }

  private async write(state: SyncLogState): Promise<void> {
    await this.io.writeFile(this.fileName, new TextEncoder().encode(JSON.stringify(state)))
  }
}

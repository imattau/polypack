import type { FileIO } from '../persistence/file-io.js'
import type { SyncOp } from './types.js'
import { syncChecksum, syncIdentityChecksum } from './checksum.js'

export interface SyncClientState {
  clientId: string
  lastAckedSeq: number
  serverCursor: number
  ops: SyncOp[]
  checksum?: string
}

/** Durable state required to resume a client without replaying acknowledged work. */
export interface SyncClientStateStore {
  load(): Promise<SyncClientState | null>
  save(state: SyncClientState): Promise<void>
}

export class MemorySyncClientStateStore implements SyncClientStateStore {
  private state: SyncClientState | null = null

  async load(): Promise<SyncClientState | null> {
    return this.state ? cloneClientState(this.state) : null
  }

  async save(state: SyncClientState): Promise<void> {
    this.state = cloneClientState(state)
  }
}

/** JSON-backed client state store for Node, OPFS, and test FileIO implementations. */
export class FileSyncClientStateStore implements SyncClientStateStore {
  private queue: Promise<unknown> = Promise.resolve()

  constructor(private readonly io: FileIO, private readonly fileName = 'sync-client-state.json') {}

  async load(): Promise<SyncClientState | null> {
    const data = await this.io.readFile(this.fileName)
    if (!data || data.length === 0) return null
    let parsed: unknown
    try {
      parsed = JSON.parse(new TextDecoder().decode(data))
    } catch {
      throw new Error('Invalid sync client state')
    }
    if (!validClientState(parsed)) throw new Error('Invalid sync client state')
    const state = parsed as SyncClientState
    if (state.checksum !== undefined && state.checksum !== syncChecksum(state.ops)) {
      throw new Error('Sync client state checksum mismatch')
    }
    return cloneClientState(state)
  }

  async save(state: SyncClientState): Promise<void> {
    await this.enqueue(async () => {
      const persisted = { ...cloneClientState(state), checksum: syncChecksum(state.ops) }
      await this.io.writeFile(this.fileName, new TextEncoder().encode(JSON.stringify(persisted)))
    })
  }

  private enqueue<T>(operation: () => Promise<T>): Promise<T> {
    const run = this.queue.then(operation)
    this.queue = run.then(() => undefined, () => undefined)
    return run
  }
}

function cloneClientState(state: SyncClientState): SyncClientState {
  return {
    clientId: state.clientId,
    lastAckedSeq: state.lastAckedSeq,
    serverCursor: state.serverCursor,
    ops: state.ops.map(op => ({ ...op, payload: structuredClone(op.payload) })),
    checksum: state.checksum,
  }
}

function validClientState(value: unknown): value is SyncClientState {
  if (!value || typeof value !== 'object') return false
  const state = value as Partial<SyncClientState>
  return typeof state.clientId === 'string' && state.clientId.length > 0 &&
    Number.isInteger(state.lastAckedSeq) && (state.lastAckedSeq ?? -1) >= 0 &&
    Number.isInteger(state.serverCursor) && (state.serverCursor ?? -1) >= 0 &&
    Array.isArray(state.ops) && state.ops.every(op => {
      if (!op || typeof op !== 'object') return false
      const candidate = op as Partial<SyncOp>
      return Number.isInteger(candidate.seq) && (candidate.seq ?? -1) >= 1 &&
        typeof candidate.timestamp === 'number' && Number.isFinite(candidate.timestamp) &&
        typeof candidate.clientId === 'string' && candidate.clientId.length > 0 &&
        typeof candidate.kind === 'string' && !!candidate.payload && typeof candidate.payload === 'object'
    })
}

export interface SyncLogState {
  baseCursor: number
  ops: SyncOp[]
  /** Identity tombstones retained after operation compaction. */
  operationIds?: string[]
  transactionIds?: string[]
  identityChecksum?: string
  /** Integrity marker for durable files; omitted by legacy logs. */
  checksum?: string
}

export interface SyncLogStats {
  baseCursor: number
  cursor: number
  operationCount: number
  operationIdentityCount: number
  transactionIdentityCount: number
}

export interface SyncOperationLog {
  load(): Promise<SyncLogState>
  stats?(): Promise<SyncLogStats>
  append(op: SyncOp): Promise<void>
  /** Persist a logical batch atomically when the adapter can provide it. */
  appendBatch?(ops: readonly SyncOp[]): Promise<void>
  compact?(baseCursor: number): Promise<void>
}

/** In-memory log useful for tests and process-local deployments. */
export class MemorySyncOperationLog implements SyncOperationLog {
  private baseCursor = 0
  private ops: SyncOp[] = []
  private operationIds = new Set<string>()
  private transactionIds = new Set<string>()

  async load(): Promise<SyncLogState> {
    return { baseCursor: this.baseCursor, ops: this.ops.map(op => ({ ...op, payload: structuredClone(op.payload) })), operationIds: [...this.operationIds], transactionIds: [...this.transactionIds] }
  }

  async stats(): Promise<SyncLogStats> {
    return { baseCursor: this.baseCursor, cursor: this.baseCursor + this.ops.length, operationCount: this.ops.length, operationIdentityCount: this.operationIds.size, transactionIdentityCount: this.transactionIds.size }
  }

  async append(op: SyncOp): Promise<void> {
    this.ops.push({ ...op, payload: structuredClone(op.payload) })
    if (op.operationId) this.operationIds.add(`${op.clientId}:${op.operationId}`)
    if (op.transactionId) this.transactionIds.add(`${op.clientId}:${op.transactionId}`)
  }

  async appendBatch(ops: readonly SyncOp[]): Promise<void> {
    for (const op of ops) await this.append(op)
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
    if (!data || data.length === 0) return { baseCursor: 0, ops: [], operationIds: [], transactionIds: [] }
    let parsed: Partial<SyncLogState>
    try {
      parsed = JSON.parse(new TextDecoder().decode(data)) as Partial<SyncLogState>
    } catch {
      throw new Error('Invalid sync operation log state')
    }
    if (!this.validState(parsed)) {
      throw new Error('Invalid sync operation log state')
    }
    if (parsed.checksum !== undefined && parsed.checksum !== syncChecksum(parsed.ops)) {
      throw new Error('Sync operation log checksum mismatch')
    }
    if (parsed.identityChecksum !== undefined && parsed.identityChecksum !== syncIdentityChecksum(parsed.operationIds ?? [], parsed.transactionIds ?? [])) {
      throw new Error('Sync operation identity checksum mismatch')
    }
    return { baseCursor: parsed.baseCursor as number, ops: parsed.ops, operationIds: parsed.operationIds, transactionIds: parsed.transactionIds, identityChecksum: parsed.identityChecksum, checksum: parsed.checksum }
  }

  async stats(): Promise<SyncLogStats> {
    const state = await this.load()
    return { baseCursor: state.baseCursor, cursor: state.baseCursor + state.ops.length, operationCount: state.ops.length, operationIdentityCount: state.operationIds?.length ?? state.ops.filter(op => op.operationId).length, transactionIdentityCount: state.transactionIds?.length ?? state.ops.filter(op => op.transactionId).length }
  }

  private validOperation(op: unknown): op is SyncOp {
    if (!op || typeof op !== 'object') return false
    const candidate = op as Partial<SyncOp>
    return Number.isInteger(candidate.seq) && (candidate.seq ?? -1) >= 1 &&
      typeof candidate.timestamp === 'number' && Number.isFinite(candidate.timestamp) &&
      typeof candidate.clientId === 'string' && candidate.clientId.length > 0 &&
      typeof candidate.kind === 'string' && !!candidate.payload && typeof candidate.payload === 'object' &&
      (candidate.operationId === undefined || (typeof candidate.operationId === 'string' && candidate.operationId.length > 0)) &&
      (candidate.transactionId === undefined || (typeof candidate.transactionId === 'string' && candidate.transactionId.length > 0)) &&
      (candidate.baseRevision === undefined || (Number.isInteger(candidate.baseRevision) && (candidate.baseRevision ?? -1) >= 0))
  }

  private validState(value: Partial<SyncLogState>): value is SyncLogState {
    if (!Number.isInteger(value.baseCursor) || (value.baseCursor ?? -1) < 0 || !Array.isArray(value.ops) || value.ops.some(op => !this.validOperation(op))) return false
    const operationKeys = new Set<string>()
    const operationIds = new Set<string>(value.operationIds ?? [])
    const transactionIds = new Set<string>(value.transactionIds ?? [])
    for (const op of value.ops) {
      const key = `${op.clientId}:${op.seq}`
      if (operationKeys.has(key)) return false
      operationKeys.add(key)
      if (op.operationId !== undefined) {
        const operationKey = `${op.clientId}:${op.operationId}`
        if (operationKeys.has(`operation:${operationKey}`)) return false
        operationKeys.add(`operation:${operationKey}`)
        operationIds.add(operationKey)
      }
      if (op.transactionId !== undefined) transactionIds.add(`${op.clientId}:${op.transactionId}`)
    }
    return [...operationIds, ...transactionIds].every(id => typeof id === 'string' && id.length > 0)
  }

  async append(op: SyncOp): Promise<void> {
    await this.appendBatch([op])
  }

  async appendBatch(ops: readonly SyncOp[]): Promise<void> {
    await this.enqueue(async () => {
      const state = await this.load()
      state.ops.push(...ops.map(op => ({ ...op, payload: structuredClone(op.payload) })))
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
    const operationIds = new Set(state.operationIds ?? [])
    const transactionIds = new Set(state.transactionIds ?? [])
    for (const op of state.ops) {
      if (op.operationId) operationIds.add(`${op.clientId}:${op.operationId}`)
      if (op.transactionId) transactionIds.add(`${op.clientId}:${op.transactionId}`)
    }
    const persisted = { baseCursor: state.baseCursor, ops: state.ops, operationIds: [...operationIds], transactionIds: [...transactionIds], identityChecksum: syncIdentityChecksum([...operationIds], [...transactionIds]), checksum: syncChecksum(state.ops) }
    await this.io.writeFile(this.fileName, new TextEncoder().encode(JSON.stringify(persisted)))
  }
}

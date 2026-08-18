import { describe, expect, it } from 'vitest'
import { MemoryFileIO } from '../src/persistence/file-io'
import { FileSyncOperationLog, SyncServer } from '../src/sync/index'

const op = (seq: number) => ({
  seq,
  timestamp: seq,
  clientId: 'client',
  kind: 'addNode' as const,
  payload: { id: `n${seq}` },
  operationId: `op-${seq}`,
})

const message = (ops: ReturnType<typeof op>[]) => ({
  type: 'delta' as const,
  clientId: 'client',
  fromSeq: 0,
  ops,
  protocolVersion: 1,
})

describe('durable sync operation logs', () => {
  it('persists accepted operations and restores the server cursor after restart', async () => {
    const io = new MemoryFileIO()
    const log = new FileSyncOperationLog(io)
    const firstMessages: unknown[] = []
    const first = new SyncServer({ operationLog: log, maxOps: 2 })
    await first.ready()
    const handle = first.addClient({ send: msg => firstMessages.push(msg) })
    handle(message([op(1), op(2), op(3)]))
    await new Promise(resolve => setTimeout(resolve, 0))
    expect(firstMessages).toHaveLength(1)
    expect(first.ops.map(item => item.operationId)).toEqual(['op-2', 'op-3'])
    expect(first.cursor).toBe(3)

    const restoredMessages: any[] = []
    const restored = new SyncServer({ operationLog: new FileSyncOperationLog(io), maxOps: 2 })
    await restored.ready()
    const restoredHandle = restored.addClient({ send: msg => restoredMessages.push(msg) })
    restoredHandle({ type: 'request-snapshot', clientId: 'late', fromSeq: 1, ops: [], protocolVersion: 1 })
    await new Promise(resolve => setTimeout(resolve, 0))
    expect(restoredMessages[0]).toMatchObject({ type: 'delta', fromSeq: 1 })
    expect(restoredMessages[0].ops.map((item: { operationId: string }) => item.operationId)).toEqual(['op-2', 'op-3'])
  })
})

import { describe, expect, it } from 'vitest'
import { MemoryFileIO } from '../src/persistence/file-io'
import { FileSyncOperationLog, SyncServer } from '../src/sync/index'
import type { SyncMessage } from '../src/sync/types'

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

  it('serializes concurrent durable submissions in cursor order', async () => {
    const io = new MemoryFileIO()
    const server = new SyncServer({ operationLog: new FileSyncOperationLog(io) })
    const received: SyncMessage[] = []
    const receive = server.addClient({ clientId: 'receiver', send: message => received.push(message) })
    const op1 = { ...op(1), clientId: 'writer-a' }
    const op2 = { ...op(1), clientId: 'writer-b', operationId: 'op-b' }
    receive({ type: 'delta', clientId: 'writer-a', fromSeq: 0, ops: [op1] })
    receive({ type: 'delta', clientId: 'writer-b', fromSeq: 0, ops: [op2] })
    await new Promise(resolve => setTimeout(resolve, 0))
    await new Promise(resolve => setTimeout(resolve, 0))
    expect(server.cursor).toBe(2)
    expect(server.ops.map(operation => operation.clientId)).toEqual(['writer-a', 'writer-b'])
    expect(received.filter(message => message.type === 'ack')).toHaveLength(2)
  })

  it('rejects a corrupted persisted operation log', async () => {
    const io = new MemoryFileIO()
    await io.writeFile('sync-operations.json', new TextEncoder().encode(JSON.stringify({
      baseCursor: 0,
      ops: [op(1)],
      checksum: 'corrupt',
    })))
    await expect(new FileSyncOperationLog(io).load()).rejects.toThrow(/checksum mismatch/)
  })

  it('rejects malformed persisted operation records', async () => {
    const io = new MemoryFileIO()
    await io.writeFile('sync-operations.json', new TextEncoder().encode(JSON.stringify({
      baseCursor: 0,
      ops: [{ seq: -1, timestamp: 'invalid', clientId: '', kind: 'addNode', payload: null }],
    })))
    await expect(new FileSyncOperationLog(io).load()).rejects.toThrow(/Invalid sync operation log state/)
  })
})

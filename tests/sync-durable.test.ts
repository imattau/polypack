import { describe, expect, it } from 'vitest'
import { MemoryFileIO } from '../src/persistence/file-io'
import { FileSyncClientStateStore, FileSyncOperationLog, MemorySyncClientStateStore, SyncServer } from '../src/sync/index'
import { PolyGraph } from '../src/graph'
import { SyncClient } from '../src/sync/client'
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

  it('rejects duplicate persisted operation identities', async () => {
    const io = new MemoryFileIO()
    await io.writeFile('sync-operations.json', new TextEncoder().encode(JSON.stringify({
      baseCursor: 0,
      ops: [op(1), op(1)],
    })))
    await expect(new FileSyncOperationLog(io).load()).rejects.toThrow(/Invalid sync operation log state/)
  })

  it('rejects corrupted durable identity tombstones', async () => {
    const io = new MemoryFileIO()
    await io.writeFile('sync-operations.json', new TextEncoder().encode(JSON.stringify({
      baseCursor: 1,
      ops: [],
      operationIds: ['client:op-1'],
      transactionIds: ['client:tx-1'],
      identityChecksum: 'corrupt',
    })))
    await expect(new FileSyncOperationLog(io).load()).rejects.toThrow(/identity checksum mismatch/)
  })

  it('flushes a durable batch before returning', async () => {
    const io = new MemoryFileIO()
    const server = new SyncServer({ operationLog: new FileSyncOperationLog(io) })
    const handle = server.addClient({ send: () => undefined })
    handle(message([op(1), op(2)]))
    await server.flush()
    const state = await new FileSyncOperationLog(io).load()
    expect(state.ops.map(item => item.operationId)).toEqual(['op-1', 'op-2'])
  })

  it('retains operation and transaction idempotency after compaction and restart', async () => {
    const io = new MemoryFileIO()
    const server = new SyncServer({ operationLog: new FileSyncOperationLog(io), maxOps: 1 })
    const sent: SyncMessage[] = []
    const handle = server.addClient({ send: message => sent.push(message) })
    const first = { ...op(1), transactionId: 'tx-1' }
    const second = { ...op(2), transactionId: 'tx-2' }
    handle(message([first]))
    handle(message([second]))
    await server.flush()
    expect(server.cursor).toBe(2)
    handle(message([first]))
    await server.flush()
    expect(server.cursor).toBe(2)

    const restored = new SyncServer({ operationLog: new FileSyncOperationLog(io), maxOps: 1 })
    await restored.ready()
    const restoredHandle = restored.addClient({ send: () => undefined })
    restoredHandle(message([first]))
    await restored.flush()
    expect(restored.cursor).toBe(2)
    expect(sent).toHaveLength(3)
  })

  it('reports durable cursor and identity retention statistics', async () => {
    const io = new MemoryFileIO()
    const server = new SyncServer({ operationLog: new FileSyncOperationLog(io), maxOps: 1 })
    const handle = server.addClient({ send: () => undefined })
    handle(message([{ ...op(1), transactionId: 'tx-1' }]))
    handle(message([{ ...op(2), transactionId: 'tx-2' }]))

    await expect(server.logStats()).resolves.toEqual({
      baseCursor: 1,
      cursor: 2,
      operationCount: 1,
      operationIdentityCount: 2,
      transactionIdentityCount: 2,
    })
  })

  it('rejects durable submissions when the pending queue is full', async () => {
    let release!: () => void
    const gate = new Promise<void>(resolve => { release = resolve })
    const log = {
      load: async () => ({ baseCursor: 0, ops: [] }),
      append: async () => undefined,
      appendBatch: async () => { await gate },
    }
    const sent: SyncMessage[] = []
    const server = new SyncServer({ operationLog: log, maxPendingOps: 1 })
    const handle = server.addClient({ send: message => sent.push(message) })
    handle(message([op(1)]))
    handle(message([op(2)]))
    expect(sent[0]).toMatchObject({ errors: [{ code: 'pending_too_large' }] })
    release()
    await server.flush()
    expect(sent).toHaveLength(2)
    expect(sent[1]).toMatchObject({ clientId: 'client' })
  })
})

describe('durable sync client state', () => {
  it('restores pending operations and cursors without replaying acknowledged work', async () => {
    const store = new MemorySyncClientStateStore()
    const state = { clientId: 'client', lastAckedSeq: 1, serverCursor: 9, ops: [op(1), op(2)] }
    await store.save(state)
    const restored = await store.load()
    expect(restored).toEqual(state)
  })

  it('persists and validates file-backed client state', async () => {
    const io = new MemoryFileIO()
    const store = new FileSyncClientStateStore(io)
    await store.save({ clientId: 'client', lastAckedSeq: 2, serverCursor: 4, ops: [op(2)] })
    expect(await store.load()).toMatchObject({ clientId: 'client', lastAckedSeq: 2, serverCursor: 4 })
    await io.writeFile('sync-client-state.json', new TextEncoder().encode(JSON.stringify({ clientId: 'client', lastAckedSeq: 0, serverCursor: 0, ops: [op(1)], checksum: 'bad' })))
    await expect(store.load()).rejects.toThrow(/checksum mismatch/)
  })

  it('restores a client and resends pending work from its durable cursor', async () => {
    const store = new MemorySyncClientStateStore()
    await store.save({ clientId: 'offline', lastAckedSeq: 0, serverCursor: 7, ops: [op(1)] })
    const sent: SyncMessage[] = []
    const client = await SyncClient.restore({
      graph: new PolyGraph(),
      clientId: 'offline',
      stateStore: store,
      retryMs: 0,
      transport: { send: message => sent.push(message), close: () => undefined, onMessage: null },
    })
    expect(sent[0]).toMatchObject({ type: 'delta', fromSeq: 0, ops: [{ operationId: 'op-1' }] })
    expect(sent[1]).toMatchObject({ type: 'request-snapshot', fromSeq: 7 })
    client.disconnect()
  })
})

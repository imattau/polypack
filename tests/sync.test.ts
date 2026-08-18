import { describe, it, expect } from 'vitest'
import { PolyGraph } from '../src/graph'
import { MemoryAdapter } from '../src/persistence/memory'
import { OpLog } from '../src/sync/oplog'
import { SyncAdapter } from '../src/sync/adapter'
import { SyncServer } from '../src/sync/server'
import { SyncClient } from '../src/sync/client'
import { MemoryTransport } from '../src/sync/transport'
import type { SyncMessage, SyncOp } from '../src/sync/types'
import type { SyncTransport } from '../src/sync/transport'
import { syncChecksum } from '../src/sync/checksum'

describe('OpLog', () => {
  it('appends ops with increasing sequence numbers', () => {
    const log = new OpLog('client-1')
    const op1 = log.append('addNode', { id: 'a' })
    const op2 = log.append('addNode', { id: 'b' })
    expect(op1.seq).toBe(1)
    expect(op2.seq).toBe(2)
    expect(op1.operationId).toBe('client-1:1')
    expect(op2.operationId).toBe('client-1:2')
    expect(log.size).toBe(2)
  })

  it('filters by sequence number', () => {
    const log = new OpLog('c1')
    log.append('addNode', { id: 'a' })
    log.append('addNode', { id: 'b' })
    log.append('addNode', { id: 'c' })
    expect(log.since(1)).toHaveLength(2)
    expect(log.since(3)).toHaveLength(0)
  })

  it('reconstructs from existing ops', () => {
    const log1 = new OpLog('c1')
    log1.append('addNode', { id: 'a' })
    log1.append('addNode', { id: 'b' })

    const log2 = new OpLog('c1', [...log1.all])
    expect(log2.size).toBe(2)
    expect(log2.latestSeq).toBe(2)
    const op3 = log2.append('addNode', { id: 'c' })
    expect(op3.seq).toBe(3)
  })

  it('checksums bigint payloads without throwing', () => {
    const op: SyncOp = { seq: 1, timestamp: 1, clientId: 'c1', kind: 'addNode', payload: { value: 1n } }
    expect(syncChecksum([op])).toBe(syncChecksum([op]))
  })
})

describe('SyncAdapter', () => {
  it('records an atomic persistence batch only after it commits', async () => {
    class RejectOnceAdapter extends MemoryAdapter {
      reject = true
      override async applyChanges(changes: Parameters<MemoryAdapter['applyChanges']>[0]): Promise<void> {
        if (this.reject) {
          this.reject = false
          throw new Error('commit failed')
        }
        await super.applyChanges(changes)
      }
    }
    const inner = new RejectOnceAdapter()
    const adapter = new SyncAdapter(inner, 'batch-client')
    const changes = {
      putNodes: [{ id: 'a', type: 't', data: {}, vector: null, insertedAt: 1, updatedAt: 1 }],
      deleteNodeIds: [], putEdges: [], deleteEdgeIds: [], putVectors: [], deleteVectorIds: [],
    }

    await expect(adapter.applyChanges(changes)).rejects.toThrow('commit failed')
    expect(adapter.oplog.size).toBe(0)
    await adapter.applyChanges(changes)
    expect(adapter.oplog.all.map(op => op.kind)).toEqual(['addNode'])
  })

  it('preserves committed transaction identity in adapter sync operations', async () => {
    const adapter = new SyncAdapter(new MemoryAdapter(), 'identity-client')
    await adapter.applyChanges({
      transactionId: 'tx-1', operationId: 'commit-1',
      putNodes: [{ id: 'a', type: 't', data: {}, vector: null, insertedAt: 1, updatedAt: 1 }],
      deleteNodeIds: [], putEdges: [], deleteEdgeIds: [], putVectors: [], deleteVectorIds: [],
    })
    expect(adapter.oplog.all[0]).toMatchObject({ transactionId: 'tx-1', operationId: 'commit-1:put-node:a' })
  })

  it('records node mutations in the op log', async () => {
    const inner = new MemoryAdapter()
    const adapter = new SyncAdapter(inner, 'client-1')

    await adapter.putNode({ id: 'n1', type: 'doc', data: {}, vector: null, insertedAt: 1, updatedAt: 1 })
    expect(adapter.oplog.size).toBe(1)
    expect(adapter.oplog.all[0].kind).toBe('addNode')

    await adapter.deleteNode('n1')
    expect(adapter.oplog.size).toBe(2)
    expect(adapter.oplog.all[1].kind).toBe('removeNode')
  })

  it('records edge mutations in the op log', async () => {
    const adapter = new SyncAdapter(new MemoryAdapter(), 'c1')
    await adapter.putEdge({ id: 'a::R::b', source: 'a', target: 'b', type: 'R', data: null, createdAt: 1 })
    expect(adapter.oplog.all[0].kind).toBe('addEdge')

    await adapter.deleteEdge('a::R::b')
    expect(adapter.oplog.all[1].kind).toBe('removeEdges')
  })

  it('records bulkDeleteEdges ops', async () => {
    const adapter = new SyncAdapter(new MemoryAdapter(), 'c1')
    await adapter.bulkPutEdges([
      { id: 'a::R::b', source: 'a', target: 'b', type: 'R', data: null, createdAt: 1 },
    ])
    await adapter.bulkDeleteEdges(['a::R::b'])
    const removeOps = adapter.oplog.all.filter(o => o.kind === 'removeEdges')
    expect(removeOps).toHaveLength(1)
    expect(removeOps[0].payload.source).toBe('a')
  })

  it('accepts independent edge IDs and rejects malformed edge records', async () => {
    const adapter = new SyncAdapter(new MemoryAdapter(), 'c1')
    await adapter.putEdge({ id: 'claim-1', source: 'a', target: 'b', type: 'R', data: null, createdAt: 1 })
    expect(adapter.oplog.all[0].payload.id).toBe('claim-1')
    await expect(adapter.putEdge({
      id: '', source: 'a', target: 'b', type: 'R', data: null, createdAt: 1,
    })).rejects.toThrow(TypeError)
  })

  it('wraps inner adapter transparently', async () => {
    const inner = new MemoryAdapter()
    const adapter = new SyncAdapter(inner, 'c1')

    await adapter.putNode({ id: 'n1', type: 'doc', data: { text: 'hello' }, vector: null, insertedAt: 1, updatedAt: 1 })
    const got = await adapter.getNode('n1')
    expect(got?.data.text).toBe('hello')
  })
})

describe('SyncServer + SyncClient', () => {
  it('can unregister a client handle', () => {
    const server = new SyncServer()
    const handle = { send: () => undefined, clientId: 'gone' }
    server.addClient(handle)

    expect(server.removeClient(handle)).toBe(true)
    expect(server.removeClient(handle)).toBe(false)
  })

  it('acknowledges operations and clears the client pending set', async () => {
    const server = new SyncServer()
    const graph = new PolyGraph()
    const { client, cleanup } = connect(graph, 'ack-client', server, false)

    graph.addNode({ id: 'x', type: 't', data: {}, insertedAt: 1, updatedAt: 1 })
    expect(client.pendingOps).toHaveLength(1)
    client.flush()
    await new Promise(resolve => setTimeout(resolve, 20))

    expect(client.pendingOps).toHaveLength(0)
    cleanup()
  })

  it('preserves transaction identity on every operation in a transaction', async () => {
    const sent: SyncMessage[] = []
    const graph = new PolyGraph()
    const client = new SyncClient({
      graph,
      clientId: 'transaction-client',
      autoFlush: false,
      retryMs: 0,
      transport: { send: message => sent.push(message), close: () => undefined, onMessage: null },
    })

    await graph.transaction(tx => {
      tx.addNode({ id: 'a', type: 't', data: {}, insertedAt: 1, updatedAt: 1 })
      tx.addNode({ id: 'b', type: 't', data: {}, insertedAt: 1, updatedAt: 1 })
    })

    const pending = client.pendingOps
    expect(pending).toHaveLength(2)
    expect(pending[0].transactionId).toBeTruthy()
    expect(pending[1].transactionId).toBe(pending[0].transactionId)
    expect(pending[0].operationId).not.toBe(pending[1].operationId)
    client.flush()
    expect(sent[0].ops.map(op => op.transactionId)).toEqual([pending[0].transactionId, pending[1].transactionId])
    client.disconnect()
  })

  it('coalesces auto-flush transaction events into one message', async () => {
    const sent: SyncMessage[] = []
    const graph = new PolyGraph()
    const client = new SyncClient({
      graph,
      clientId: 'coalesced-transaction',
      retryMs: 0,
      transport: { send: message => sent.push(message), close: () => undefined, onMessage: null },
    })
    await graph.transaction(tx => {
      tx.addNode({ id: 'a', type: 't', data: {}, insertedAt: 1, updatedAt: 1 })
      tx.addNode({ id: 'b', type: 't', data: {}, insertedAt: 1, updatedAt: 1 })
    })
    await new Promise(resolve => queueMicrotask(resolve))

    expect(sent).toHaveLength(1)
    expect(sent[0].ops).toHaveLength(2)
    expect(sent[0].ops[0].transactionId).toBe(sent[0].ops[1].transactionId)
    client.disconnect()
  })

  it('deduplicates retried operations while acknowledging every delivery', () => {
    const server = new SyncServer()
    const acknowledgements: SyncMessage[] = []
    const broadcasts: SyncMessage[] = []
    const sender = { send: (msg: SyncMessage) => acknowledgements.push(msg), clientId: 'sender' }
    const receive = server.addClient(sender)
    server.addClient({ send: (msg) => broadcasts.push(msg), clientId: 'receiver' })
    const op = { seq: 1, clientId: 'sender', timestamp: 1, kind: 'addNode' as const, payload: { id: 'x' } }
    const delta: SyncMessage = { type: 'delta', clientId: 'sender', fromSeq: 0, ops: [op] }

    receive(delta)
    receive(delta)

    expect(server.ops).toHaveLength(1)
    expect(broadcasts).toHaveLength(1)
    expect(acknowledgements).toHaveLength(2)
    expect(acknowledgements[1]).toMatchObject({ type: 'ack', clientId: 'sender', fromSeq: 1 })
  })

  it('deduplicates logical operations by operation ID across sequence changes', () => {
    const server = new SyncServer()
    const acknowledgements: SyncMessage[] = []
    const receive = server.addClient({ clientId: 'sender', send: message => acknowledgements.push(message) })
    const operation = (seq: number): SyncOp => ({
      seq, timestamp: 1, clientId: 'sender', operationId: 'logical-1', kind: 'addNode',
      payload: { id: 'logical', type: 't', data: {}, insertedAt: 1, updatedAt: 1 },
    })
    receive({ type: 'delta', clientId: 'sender', fromSeq: 0, ops: [operation(1)] })
    receive({ type: 'delta', clientId: 'sender', fromSeq: 1, ops: [operation(2)] })

    expect(server.ops).toHaveLength(1)
    expect(acknowledgements[1]).toMatchObject({ type: 'ack', fromSeq: 2 })
  })

  it('does not deduplicate equal operation IDs from different clients', () => {
    const server = new SyncServer()
    const receiveA = server.addClient({ clientId: 'a', send: () => undefined })
    const receiveB = server.addClient({ clientId: 'b', send: () => undefined })
    const op = (clientId: string, seq: number): SyncOp => ({
      seq, timestamp: 1, clientId, operationId: 'same-local-id', kind: 'addNode',
      payload: { id: `${clientId}-node`, type: 't', data: {}, insertedAt: 1, updatedAt: 1 },
    })
    receiveA({ type: 'delta', clientId: 'a', fromSeq: 0, ops: [op('a', 1)] })
    receiveB({ type: 'delta', clientId: 'b', fromSeq: 0, ops: [op('b', 1)] })
    expect(server.ops).toHaveLength(2)
  })

  it('bounds deduplication memory with operation-log compaction', () => {
    const server = new SyncServer({ maxOps: 1 })
    const receive = server.addClient({ clientId: 'compact-client', send: () => undefined })
    const op = (seq: number): SyncOp => ({
      seq, timestamp: 1, clientId: 'compact-client', operationId: `op-${seq}`, kind: 'addNode',
      payload: { id: `node-${seq}`, type: 't', data: {}, insertedAt: 1, updatedAt: 1 },
    })
    receive({ type: 'delta', clientId: 'compact-client', fromSeq: 0, ops: [op(1)] })
    receive({ type: 'delta', clientId: 'compact-client', fromSeq: 1, ops: [op(2)] })
    expect(server.ops.map(item => item.operationId)).toEqual(['op-2'])
  })

  it('retries an unacknowledged operation until an acknowledgement arrives', async () => {
    const graph = new PolyGraph()
    let sends = 0
    const transport: SyncTransport = {
      onMessage: null,
      close: () => undefined,
      send: (msg) => {
        sends++
        if (sends === 2) {
          setTimeout(() => transport.onMessage?.({
            type: 'ack', clientId: msg.clientId, fromSeq: msg.ops[0].seq, ops: [],
          }), 0)
        }
      },
    }
    const client = new SyncClient({ graph, transport, clientId: 'retry-client', retryMs: 5 })

    graph.addNode({ id: 'retry', type: 't', data: {}, insertedAt: 1, updatedAt: 1 })
    await new Promise(resolve => setTimeout(resolve, 30))

    expect(sends).toBe(2)
    expect(client.pendingOps).toHaveLength(0)
    client.disconnect()
  })

  it('bounds retained acknowledged client operations', () => {
    const sent: SyncMessage[] = []
    const graph = new PolyGraph()
    const client = new SyncClient({
      graph, clientId: 'retained-client', autoFlush: false, retryMs: 0, maxRetainedOps: 1,
      transport: { send: message => sent.push(message), close: () => undefined, onMessage: null },
    })
    graph.addNode({ id: 'a', type: 't', data: {}, insertedAt: 1, updatedAt: 1 })
    graph.addNode({ id: 'b', type: 't', data: {}, insertedAt: 1, updatedAt: 1 })
    client.flush()
    sent[0].ops.length
    client.handleMessage({ type: 'ack', clientId: 'retained-client', fromSeq: 2, ops: [] })

    expect(client.pendingOps).toHaveLength(0)
    expect(client.oplog.size).toBe(1)
    client.disconnect()
  })

  it('resends pending operations after reconnecting with a new transport', () => {
    const graph = new PolyGraph()
    const firstMessages: SyncMessage[] = []
    const secondMessages: SyncMessage[] = []
    const first: SyncTransport = { onMessage: null, close: () => undefined, send: msg => firstMessages.push(msg) }
    const second: SyncTransport = { onMessage: null, close: () => undefined, send: msg => secondMessages.push(msg) }
    const client = new SyncClient({ graph, transport: first, clientId: 'reconnect-client', autoFlush: false, retryMs: 0 })

    graph.addNode({ id: 'pending', type: 't', data: {}, insertedAt: 1, updatedAt: 1 })
    client.flush()
    client.reconnect(second)

    expect(firstMessages).toHaveLength(1)
    expect(secondMessages.map(msg => msg.type)).toEqual(['delta', 'request-snapshot'])
    second.onMessage?.({ type: 'ack', clientId: 'reconnect-client', fromSeq: 1, ops: [] })
    expect(client.pendingOps).toHaveLength(0)
    client.disconnect()
  })

  it('flushes buffered ops on disconnect instead of discarding them', () => {
    const graph = new PolyGraph()
    const messages: SyncMessage[] = []
    const transport: SyncTransport = { onMessage: null, close: () => undefined, send: msg => messages.push(msg) }
    const client = new SyncClient({ graph, transport, clientId: 'disconnect-client', autoFlush: false, retryMs: 0 })

    graph.addNode({ id: 'unflushed', type: 't', data: {}, insertedAt: 1, updatedAt: 1 })
    expect(client.pendingOps).toHaveLength(1)
    expect(messages).toHaveLength(0)

    client.disconnect()

    expect(messages).toHaveLength(1)
    expect(messages[0].type).toBe('delta')
  })

  it('catches up a late client from a server snapshot', async () => {
    const server = new SyncServer()
    const sourceGraph = new PolyGraph()
    const { client: source, cleanup: cleanupSource } = connect(sourceGraph, 'source', server, false)
    sourceGraph.addNode({ id: 'existing', type: 'note', data: { text: 'before' }, insertedAt: 1, updatedAt: 1 })
    source.flush()
    await new Promise(resolve => setTimeout(resolve, 10))

    const lateGraph = new PolyGraph()
    const { client: late, cleanup: cleanupLate } = connect(lateGraph, 'late', server, false)
    late.requestSync()
    await new Promise(resolve => setTimeout(resolve, 10))

    expect(lateGraph.getNode('existing')?.data.text).toBe('before')
    expect(late.syncCursor).toBe(server.cursor)
    cleanupSource()
    cleanupLate()
  })

  it('detects a server cursor gap and requests recovery', async () => {
    const server = new SyncServer()
    const receive = server.addClient({ send: () => undefined, clientId: 'seed' })
    const makeNodeOp = (seq: number, id: string) => ({
      seq,
      clientId: 'seed',
      timestamp: seq,
      kind: 'addNode' as const,
      payload: { id, type: 't', data: {}, vector: null, insertedAt: seq, updatedAt: seq },
    })
    const first = makeNodeOp(1, 'first')
    const second = makeNodeOp(2, 'second')
    receive({ type: 'delta', clientId: 'seed', fromSeq: 0, ops: [first, second] })

    const graph = new PolyGraph()
    const { client, cleanup } = connect(graph, 'recovering', server, false)
    client.handleMessage({ type: 'delta', clientId: 'server', fromSeq: 1, ops: [second] })
    await new Promise(resolve => setTimeout(resolve, 10))

    expect(graph.getNode('first')).toBeDefined()
    expect(graph.getNode('second')).toBeDefined()
    expect(client.syncCursor).toBe(2)
    cleanup()
  })

  it('returns only operations after a requested server cursor', () => {
    const server = new SyncServer()
    const seedReceive = server.addClient({ send: () => undefined, clientId: 'seed' })
    const ops = ['a', 'b'].map((id, index) => ({
      seq: index + 1,
      clientId: 'seed',
      timestamp: index,
      kind: 'removeNode' as const,
      payload: { id },
    }))
    seedReceive({ type: 'delta', clientId: 'seed', fromSeq: 0, ops })
    const responses: SyncMessage[] = []
    const request = server.addClient({ send: msg => responses.push(msg), clientId: 'late' })

    request({ type: 'request-snapshot', clientId: 'late', fromSeq: 1, ops: [] })

    expect(responses).toHaveLength(1)
    expect(responses[0]).toMatchObject({ type: 'delta', clientId: 'server', fromSeq: 1 })
    expect(responses[0].ops.map(op => op.payload.id)).toEqual(['b'])
  })

  it('falls back to a full snapshot when a recovery cursor is invalid', () => {
    const server = new SyncServer()
    const seedReceive = server.addClient({ send: () => undefined, clientId: 'seed' })
    seedReceive({
      type: 'delta', clientId: 'seed', fromSeq: 0, ops: [{
        seq: 1, clientId: 'seed', timestamp: 1, kind: 'removeNode', payload: { id: 'a' },
      }],
    })
    const responses: SyncMessage[] = []
    const request = server.addClient({ send: msg => responses.push(msg), clientId: 'late' })

    request({ type: 'request-snapshot', clientId: 'late', fromSeq: 99, ops: [] })

    expect(responses[0]).toMatchObject({ type: 'snapshot', clientId: 'server', fromSeq: 0 })
    expect(responses[0].ops).toHaveLength(1)
  })

  /** Wire a client to the server. The server broadcasts to all other clients'
   *  handleMessage directly (not through the transport). */
  function connect(
    graph: PolyGraph,
    clientId: string,
    server: SyncServer,
    autoFlush = true,
  ): { client: SyncClient; cleanup: () => void } {
    const [clientT, serverT] = MemoryTransport.pair()
    const client = new SyncClient({ graph, transport: clientT, clientId, autoFlush })

    // Register with server: broadcast messages go to client.handleMessage directly
    const onServerMsg = server.addClient({
      send: (msg) => client.handleMessage(msg),
      clientId,
    })

    // Client → Server direction: transport delivers to server handler
    serverT.onMessage = (msg) => onServerMsg(msg)

    return {
      client,
      cleanup: () => client.disconnect(),
    }
  }

  it('syncs a mutation from one client to another', async () => {
    const server = new SyncServer()
    const aGraph = new PolyGraph()
    const bGraph = new PolyGraph()

    const { client: alice, cleanup: ca } = connect(aGraph, 'alice', server)
    const { client: bob, cleanup: cb } = connect(bGraph, 'bob', server)

    aGraph.addNode({ id: 'shared', type: 'note', data: { text: 'hello' }, insertedAt: 1, updatedAt: 1 })
    await new Promise(r => setTimeout(r, 30))

    expect(bGraph.getNode('shared')).toBeDefined()
    expect(bGraph.getNode('shared')?.data.text).toBe('hello')
    expect(server.ops).toHaveLength(1)

    ca()
    cb()
  })

  it('does not echo local changes back', async () => {
    const [clientT, serverT] = MemoryTransport.pair()
    const graph = new PolyGraph()
    const client = new SyncClient({ graph, transport: clientT, clientId: 'echo-test' })

    const onServerMsg = new SyncServer().addClient({
      send: (msg) => client.handleMessage(msg),
    })
    serverT.onMessage = (msg) => onServerMsg(msg)

    const changes: string[] = []
    const sub = graph.changes.subscribe(e => changes.push(e.type))

    graph.addNode({ id: 'x', type: 't', data: {}, insertedAt: 1, updatedAt: 1 })
    // Give time for flush + echo back
    await new Promise(r => setTimeout(r, 30))

    expect(graph.size).toBe(1)
    // Node should only be added once
    expect(changes.filter(c => c === 'node_added')).toHaveLength(1)

    sub.unsubscribe()
    client.disconnect()
  })

  it('does not replay a client\'s own ops during cursor catch-up', async () => {
    const server = new SyncServer()
    const aGraph = new PolyGraph()
    const bGraph = new PolyGraph()

    const { client: alice, cleanup: ca } = connect(aGraph, 'alice', server, false)
    const { client: bob, cleanup: cb } = connect(bGraph, 'bob', server, false)

    const aEvents: string[] = []
    const sub = aGraph.changes.subscribe(e => { if (e.nodeId === 'n1') aEvents.push(e.type) })

    // Alice contributes first; her cursor never advances past her own op until
    // a peer broadcasts, which triggers a gap catch-up that re-delivers it.
    aGraph.addNode({ id: 'n1', type: 't', data: {}, insertedAt: 1, updatedAt: 1 })
    alice.flush()
    await new Promise(resolve => setTimeout(resolve, 10))

    bGraph.addNode({ id: 'n2', type: 't', data: {}, insertedAt: 2, updatedAt: 2 })
    bob.flush()
    await new Promise(resolve => setTimeout(resolve, 10))

    // Catch-up must deliver Bob's op without re-applying Alice's own.
    alice.requestSync()
    await new Promise(resolve => setTimeout(resolve, 10))

    expect(aGraph.getNode('n2')).toBeDefined()
    expect(aEvents.filter(e => e === 'node_added')).toHaveLength(1)
    expect(alice.syncCursor).toBe(server.cursor)

    sub.unsubscribe()
    ca()
    cb()
  })

  it('syncs cascade deletion across clients', async () => {
    const server = new SyncServer()
    const aGraph = new PolyGraph()
    const bGraph = new PolyGraph()

    const { client: alice, cleanup: ca } = connect(aGraph, 'alice', server, false)
    const { client: bob, cleanup: cb } = connect(bGraph, 'bob', server, false)

    aGraph.addNode({ id: 'parent', type: 't', data: {}, insertedAt: 1, updatedAt: 1 })
    aGraph.addNode({ id: 'child', type: 't', data: {}, insertedAt: 2, updatedAt: 2 })
    aGraph.addEdge('parent', 'OWNS', 'child', undefined, 'owned')
    alice.flush()
    await new Promise(r => setTimeout(r, 10))

    expect(bGraph.getNode('parent')).toBeDefined()
    expect(bGraph.getNode('child')).toBeDefined()

    aGraph.removeNode('parent')
    alice.flush()
    await new Promise(r => setTimeout(r, 10))

    expect(bGraph.getNode('parent')).toBeUndefined()
    expect(bGraph.getNode('child')).toBeUndefined()

    ca()
    cb()
  })

  it('handles concurrent mutations from two clients', async () => {
    const server = new SyncServer()
    const aGraph = new PolyGraph()
    const bGraph = new PolyGraph()

    const { client: alice, cleanup: ca } = connect(aGraph, 'alice', server, false)
    const { client: bob, cleanup: cb } = connect(bGraph, 'bob', server, false)

    aGraph.addNode({ id: 'from-alice', type: 't', data: {}, insertedAt: 1, updatedAt: 1 })
    bGraph.addNode({ id: 'from-bob', type: 't', data: {}, insertedAt: 2, updatedAt: 2 })

    alice.flush()
    bob.flush()
    await new Promise(r => setTimeout(r, 30))

    expect(aGraph.getNode('from-bob')).toBeDefined()
    expect(bGraph.getNode('from-alice')).toBeDefined()
    expect(server.ops).toHaveLength(2)

    ca()
    cb()
  })

  it('syncs node updates across clients', async () => {
    const server = new SyncServer()
    const aGraph = new PolyGraph()
    const bGraph = new PolyGraph()

    const { client: alice, cleanup: ca } = connect(aGraph, 'alice', server, false)
    const { client: bob, cleanup: cb } = connect(bGraph, 'bob', server, false)

    aGraph.addNode({ id: 'n', type: 't', data: { val: 1 }, insertedAt: 1, updatedAt: 1 })
    alice.flush()
    await new Promise(r => setTimeout(r, 10))

    aGraph.updateNode('n', { val: 2 })
    alice.flush()
    await new Promise(r => setTimeout(r, 10))

    expect((bGraph.getNode('n')?.data as any).val).toBe(2)
    ca()
    cb()
  })

  it('syncs edge removal across clients', async () => {
    const server = new SyncServer()
    const aGraph = new PolyGraph()
    const bGraph = new PolyGraph()

    const { client: alice, cleanup: ca } = connect(aGraph, 'alice', server, false)
    const { client: bob, cleanup: cb } = connect(bGraph, 'bob', server, false)

    aGraph.addNode({ id: 'a', type: 't', data: {}, insertedAt: 1, updatedAt: 1 })
    aGraph.addNode({ id: 'b', type: 't', data: {}, insertedAt: 2, updatedAt: 2 })
    aGraph.addEdge('a', 'REL', 'b')
    alice.flush()
    await new Promise(r => setTimeout(r, 10))

    aGraph.removeEdges('a', 'REL', 'b')
    alice.flush()
    await new Promise(r => setTimeout(r, 10))

    expect(bGraph.getEdgeTargets('a', 'REL')).toHaveLength(0)
    ca()
    cb()
  })

  it('encodes edge updates distinctly from edge additions', () => {
    const graph = new PolyGraph()
    graph.addNode({ id: 'a', type: 't', data: {}, insertedAt: 1, updatedAt: 1 })
    graph.addNode({ id: 'b', type: 't', data: {}, insertedAt: 1, updatedAt: 1 })
    graph.addEdge({ id: 'claim', source: 'a', target: 'b', type: 'REL', data: { state: 'new' }, createdAt: 1 })
    const sent: SyncMessage[] = []
    const client = new SyncClient({
      graph,
      transport: { send: message => sent.push(message), close: () => undefined, onMessage: null },
      autoFlush: false,
      clientId: 'edge-updater',
    })
    graph.updateEdge('claim', { state: 'reviewed' }, { expectedRevision: 0 })
    client.flush()
    expect(sent[0].ops.at(-1)?.kind).toBe('updateEdge')
    client.disconnect()
  })

  it('server onOp callback fires for each received op', async () => {
    const received: string[] = []
    const server = new SyncServer()
    server.onOp = (op) => received.push(op.kind)

    const graph = new PolyGraph()
    const { client, cleanup } = connect(graph, 'tester', server, false)

    graph.addNode({ id: 'x', type: 't', data: {}, insertedAt: 1, updatedAt: 1 })
    client.flush()
    await new Promise(r => setTimeout(r, 10))

    expect(received).toEqual(['addNode'])
    cleanup()
  })

  it('supports authorization hooks and reports rejected operations', async () => {
    const responses: SyncMessage[] = []
    const server = new SyncServer({ authorize: async op => op.payload.allowed === true })
    const sender = { clientId: 'authorized-client', send: (message: SyncMessage) => responses.push(message) }
    const receive = server.addClient(sender)
    receive({
      type: 'delta', clientId: 'authorized-client', fromSeq: 0, ops: [
        { seq: 1, timestamp: 1, clientId: 'authorized-client', kind: 'addNode', payload: { id: 'denied' } },
        { seq: 2, timestamp: 1, clientId: 'authorized-client', kind: 'addNode', payload: { id: 'accepted', allowed: true } },
      ],
    })
    await new Promise(resolve => setTimeout(resolve, 0))
    expect(server.ops.map(op => op.payload.id)).toEqual(['accepted'])
    expect(responses[0].errors?.[0].code).toBe('unauthorized')
  })

  it('reports base-revision conflicts through the server hook', async () => {
    const responses: SyncMessage[] = []
    const errors: string[] = []
    const server = new SyncServer({
      conflict: op => op.baseRevision === 2 ? { ok: true } : { ok: false, message: 'stale revision' },
    })
    const client = new SyncClient({
      graph: new PolyGraph(),
      transport: { send: () => undefined, close: () => undefined, onMessage: null },
      clientId: 'conflict-client',
      onError: error => errors.push(error.code),
    })
    const sender = { clientId: 'conflict-client', send: (message: SyncMessage) => { responses.push(message); client.handleMessage(message) } }
    const receive = server.addClient(sender)
    receive({
      type: 'delta', clientId: 'conflict-client', fromSeq: 0, ops: [
        { seq: 1, timestamp: 1, clientId: 'conflict-client', kind: 'updateNode', baseRevision: 1, payload: { id: 'n' } },
      ],
    })
    await new Promise(resolve => setTimeout(resolve, 0))
    expect(server.ops).toHaveLength(0)
    expect(responses[0].errors).toMatchObject([{ code: 'conflict', message: 'stale revision' }])
    expect(errors).toEqual(['conflict'])
    client.disconnect()
  })

  it('filters subscriptions and expires cursors after bounded compaction', () => {
    const server = new SyncServer({ maxOps: 1 })
    const received: SyncMessage[] = []
    const sender = { clientId: 'sender', send: () => undefined }
    const receiver = { clientId: 'receiver', send: (message: SyncMessage) => received.push(message) }
    const send = server.addClient(sender)
    server.addClient(receiver, { filter: op => op.kind === 'addNode' })
    const op = (seq: number, kind: SyncOp['kind']) => ({ seq, timestamp: 1, clientId: 'sender', kind, payload: { id: `${kind}-${seq}` } })
    send({ type: 'delta', clientId: 'sender', fromSeq: 0, ops: [op(1, 'addNode'), op(2, 'removeNode')] })
    expect(server.cursor).toBe(2)
    expect(received[0].ops.map(item => item.kind)).toEqual(['addNode'])

    const recovery: SyncMessage[] = []
    server.addClient({ clientId: 'recovery', send: message => recovery.push(message) })({ type: 'request-snapshot', clientId: 'recovery', fromSeq: 0, ops: [] })
    expect(recovery[0].errors?.[0].code).toBe('cursor_expired')
  })

  it('advances filtered clients using the global server cursor', () => {
    const server = new SyncServer()
    const graph = new PolyGraph()
    const client = new SyncClient({
      graph,
      clientId: 'filtered-client',
      retryMs: 0,
      transport: { send: () => undefined, close: () => undefined, onMessage: null },
    })
    const sender = { clientId: 'sender', send: () => undefined }
    const received: SyncMessage[] = []
    const receiver = { clientId: 'filtered-client', send: (message: SyncMessage) => received.push(message) }
    const send = server.addClient(sender)
    server.addClient(receiver, { filter: op => op.kind === 'addNode' })

    send({ type: 'delta', clientId: 'sender', fromSeq: 0, ops: [
      { seq: 1, timestamp: 1, clientId: 'sender', kind: 'addNode', payload: { id: 'visible', type: 't', data: {}, insertedAt: 1, updatedAt: 1 } },
      { seq: 2, timestamp: 1, clientId: 'sender', kind: 'removeNode', payload: { id: 'hidden' } },
    ] })
    client.handleMessage(received[0])

    expect(received[0].cursor).toBe(2)
    expect(client.syncCursor).toBe(2)
    client.disconnect()
  })

  it('bounds sync batches and splits client flushes', async () => {
    const responses: SyncMessage[] = []
    const server = new SyncServer({ maxBatchOps: 1 })
    const receive = server.addClient({ clientId: 'bounded', send: message => responses.push(message) })
    receive({ type: 'delta', clientId: 'bounded', fromSeq: 0, ops: [
      { seq: 1, timestamp: 1, clientId: 'bounded', kind: 'addNode', payload: { id: 'a' } },
      { seq: 2, timestamp: 1, clientId: 'bounded', kind: 'addNode', payload: { id: 'b' } },
    ], protocolVersion: 1 })
    expect(responses[0].errors?.[0].code).toBe('batch_too_large')

    const sent: SyncMessage[] = []
    const graph = new PolyGraph()
    const client = new SyncClient({
      graph,
      clientId: 'split',
      autoFlush: false,
      maxOpsPerMessage: 1,
      transport: { send: message => sent.push(message), close: () => undefined, onMessage: null },
    })
    graph.addNode({ id: 'a', type: 't', data: {}, insertedAt: 1, updatedAt: 1 })
    graph.addNode({ id: 'b', type: 't', data: {}, insertedAt: 1, updatedAt: 1 })
    client.flush()
    expect(sent).toHaveLength(1)
    expect(sent[0].ops).toHaveLength(1)
    client.disconnect()
  })

  it('paginates oversized server recovery snapshots', () => {
    const server = new SyncServer({ maxBatchOps: 1 })
    const sent: SyncMessage[] = []
    const sender = { clientId: 'sender', send: () => undefined }
    const request = server.addClient(sender)
    request({ type: 'delta', clientId: 'sender', fromSeq: 0, ops: [
      { seq: 1, timestamp: 1, clientId: 'sender', kind: 'addNode', payload: { id: 'a' } },
    ] })
    request({ type: 'delta', clientId: 'sender', fromSeq: 1, ops: [
      { seq: 2, timestamp: 1, clientId: 'sender', kind: 'addNode', payload: { id: 'b' } },
    ] })
    const recovery = server.addClient({ clientId: 'recovery', send: message => sent.push(message) })

    recovery({ type: 'request-snapshot', clientId: 'recovery', fromSeq: 0, ops: [] })
    expect(sent[0]).toMatchObject({ type: 'snapshot', cursor: 1, more: true })
    expect(sent[0].ops).toHaveLength(1)

    recovery({ type: 'request-snapshot', clientId: 'recovery', fromSeq: sent[0].cursor, ops: [] })
    expect(sent[1]).toMatchObject({ type: 'delta', cursor: 2, more: false })
    expect(sent[1].ops[0].payload).toMatchObject({ id: 'b' })
  })

  it('detects corrupted server deltas and requests recovery', () => {
    const requested: SyncMessage[] = []
    const client = new SyncClient({
      graph: new PolyGraph(),
      clientId: 'checksum-client',
      onError: error => expect(error.code).toBe('checksum_mismatch'),
      transport: { send: message => requested.push(message), close: () => undefined, onMessage: null },
    })
    client.handleMessage({
      type: 'delta', clientId: 'server', fromSeq: 0,
      ops: [{ seq: 1, timestamp: 1, clientId: 'server', kind: 'addNode', payload: { id: 'n' } }],
      checksum: 'corrupt', protocolVersion: 1,
    })
    expect(requested[0]).toMatchObject({ type: 'request-snapshot', fromSeq: 0 })
    client.disconnect()
  })
})

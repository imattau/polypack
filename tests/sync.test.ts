import { describe, it, expect } from 'vitest'
import { PolyGraph } from '../src/graph'
import { MemoryAdapter } from '../src/persistence/memory'
import { OpLog } from '../src/sync/oplog'
import { SyncAdapter } from '../src/sync/adapter'
import { SyncServer } from '../src/sync/server'
import { SyncClient } from '../src/sync/client'
import { MemoryTransport } from '../src/sync/transport'
import type { SyncMessage } from '../src/sync/types'
import type { SyncTransport } from '../src/sync/transport'

describe('OpLog', () => {
  it('appends ops with increasing sequence numbers', () => {
    const log = new OpLog('client-1')
    const op1 = log.append('addNode', { id: 'a' })
    const op2 = log.append('addNode', { id: 'b' })
    expect(op1.seq).toBe(1)
    expect(op2.seq).toBe(2)
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
})

describe('SyncAdapter', () => {
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

  it('rejects edge records whose IDs cannot be reconstructed safely', async () => {
    const adapter = new SyncAdapter(new MemoryAdapter(), 'c1')
    await expect(adapter.putEdge({
      id: 'wrong', source: 'a', target: 'b', type: 'R', data: null, createdAt: 1,
    })).rejects.toThrow('Invalid edge ID')
    await expect(adapter.putEdge({
      id: 'a::part::R::b', source: 'a::part', target: 'b', type: 'R', data: null, createdAt: 1,
    })).rejects.toThrow(RangeError)
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
    expect(secondMessages).toHaveLength(1)
    second.onMessage?.({ type: 'ack', clientId: 'reconnect-client', fromSeq: 1, ops: [] })
    expect(client.pendingOps).toHaveLength(0)
    client.disconnect()
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
})

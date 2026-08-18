import { describe, expect, it } from 'vitest'
import { ConflictError, MemoryAdapter, PolyGraph } from '../src/index'
import { BinaryStoreAdapter } from '../src/persistence/binary-store'
import { MemoryFileIO } from '../src/persistence/file-io'

const node = (id: string, data: Record<string, unknown> = {}) => ({
  id, type: 'record', data, insertedAt: 1, updatedAt: 1,
})

describe('database-core mutation API', () => {
  it('commits a transaction atomically and emits after persistence', async () => {
    const adapter = new MemoryAdapter()
    const graph = new PolyGraph(adapter)
    const events: string[] = []
    graph.changes.subscribe(event => events.push(event.type))

    await graph.transaction(async tx => {
      tx.addNode(node('a'))
      expect(tx.getNode('a')?.data).toEqual({})
      tx.addNode(node('b'))
      tx.addEdge({ id: 'a::LINKS::b', source: 'a', target: 'b', type: 'LINKS', createdAt: 1, revision: 4 })
      expect(graph.getNode('b')).toBeDefined()
    })

    expect(await adapter.getNode('a')).toBeDefined()
    expect(events).toEqual(['node_added', 'node_added', 'edge_added'])
  })

  it('rolls back memory, persistence queue, and events when the callback fails', async () => {
    const graph = new PolyGraph(new MemoryAdapter())
    const events: string[] = []
    graph.changes.subscribe(event => events.push(event.type))

    await expect(graph.transaction(async tx => {
      tx.addNode(node('temporary'))
      throw new Error('abort')
    })).rejects.toThrow('abort')

    expect(graph.getNode('temporary')).toBeUndefined()
    expect(events).toEqual([])
    await graph.flush()
    expect(await graph.persistence.getNode('temporary')).toBeUndefined()
  })

  it('rejects nested transactions', async () => {
    const graph = new PolyGraph()
    await expect(graph.transaction(() => graph.transaction(() => undefined))).rejects.toThrow('Nested transactions')
  })

  it('blocks interleaved ordinary mutations during an async transaction', async () => {
    const graph = new PolyGraph()
    await expect(graph.transaction(async tx => {
      tx.addNode(node('inside'))
      await Promise.resolve()
      expect(() => graph.addNode(node('outside'))).toThrow('transaction context')
    })).resolves.toBeUndefined()
  })

  it('increments revisions and rejects stale conditional writes', () => {
    const graph = new PolyGraph()
    graph.addNode(node('n', { name: 'Alice' }))
    expect(graph.getNode('n')?.revision).toBe(0)

    graph.updateNode('n', { name: 'Bob' }, { expectedRevision: 0 })
    expect(graph.getNode('n')?.revision).toBe(1)
    expect(() => graph.updateNode('n', { name: 'Carol' }, { expectedRevision: 0 }))
      .toThrow(ConflictError)
    expect(() => graph.removeNode('n', { expectedRevision: 0 })).toThrow(ConflictError)
  })

  it('applies nested set, unset, increment, and compare-and-set patches', () => {
    const graph = new PolyGraph()
    graph.addNode(node('n', { profile: { name: 'Alice', temporary: true }, views: 2 }))
    const updated = graph.patchNode('n', {
      set: { 'data.profile.name': 'Mary Smith' },
      unset: ['data.profile.temporary'],
      increment: { 'data.views': 1 },
      compareAndSet: { 'data.profile.name': { expected: 'Alice', value: 'Mary Smith' } },
    }, { expectedRevision: 0 })

    expect(updated?.data).toEqual({ profile: { name: 'Mary Smith' }, views: 3 })
    expect(updated?.revision).toBe(1)
  })

  it('supports conditional edge removal', async () => {
    const graph = new PolyGraph()
    graph.addNode(node('a'))
    graph.addNode(node('b'))
    graph.addEdge({ id: 'a::LINKS::b', source: 'a', target: 'b', type: 'LINKS', createdAt: 1, revision: 3 })

    await expect(graph.transaction(tx => tx.removeEdge('a::LINKS::b', { expectedRevision: 2 })))
      .rejects.toThrow(ConflictError)
    expect(graph.getEdgeTargets('a', 'LINKS')).toEqual(['b'])
    await graph.transaction(tx => { expect(tx.removeEdge('a::LINKS::b', { expectedRevision: 3 })).toBe(true) })
    expect(graph.getEdgeTargets('a', 'LINKS')).toEqual([])
  })

  it('persists parallel edges with independent IDs and reloads them', async () => {
    const io = new MemoryFileIO()
    const adapter = new BinaryStoreAdapter({ storeDir: 'parallel-edge-test', fileIO: io })
    const graph = new PolyGraph(adapter)
    graph.addNode(node('a'))
    graph.addNode(node('b'))
    graph.addEdge({ id: 'claim-1', source: 'a', target: 'b', type: 'CLAIMS', createdAt: 1 })
    graph.addEdge({ id: 'claim-2', source: 'a', target: 'b', type: 'CLAIMS', createdAt: 2 })
    await graph.flush()

    expect(graph.getEdges('a', 'CLAIMS').map(edge => edge.id).sort()).toEqual(['claim-1', 'claim-2'])
    const reloaded = new PolyGraph(new BinaryStoreAdapter({ storeDir: 'parallel-edge-test', fileIO: io }))
    await reloaded.warm()
    expect(reloaded.getEdges('a', 'CLAIMS').map(edge => edge.id).sort()).toEqual(['claim-1', 'claim-2'])
  })

  it('records acknowledged logical mutations with durable sequences', async () => {
    const io = new MemoryFileIO()
    const adapter = new BinaryStoreAdapter({ storeDir: 'mutation-log-test', fileIO: io })
    const graph = new PolyGraph(adapter)
    let transactionId = ''
    await graph.transaction(tx => {
      transactionId = tx.id
      tx.addNode(node('logged'))
    })

    const records = await adapter.getMutationsSince!(0n)
    expect(records).toHaveLength(1)
    expect(records[0].sequence).toBe(1n)
    expect(records[0].transactionId).toBe(transactionId)
    expect(records[0].operations[0].type).toBe('putNode')

    const reopened = new BinaryStoreAdapter({ storeDir: 'mutation-log-test', fileIO: io })
    expect(await reopened.latestMutationSequence!()).toBe(1n)
    expect((await reopened.getMutationsSince!(0n))[0].operationId).toBe(records[0].operationId)
  })

  it('deduplicates repeated operation IDs in the mutation log', async () => {
    const adapter = new MemoryAdapter()
    const changes = {
      operationId: 'retry-1', transactionId: 'tx-retry',
      putNodes: [{ ...node('once'), vector: null, revision: 0 }],
      deleteNodeIds: [], putEdges: [], deleteEdgeIds: [], putVectors: [], deleteVectorIds: [],
    }
    await adapter.applyChanges(changes)
    await adapter.applyChanges(changes)
    expect(await adapter.latestMutationSequence!()).toBe(1n)
    expect((await adapter.getMutationsSince!(0n))).toHaveLength(1)
  })

  it('keeps snapshot queries isolated from later writes', async () => {
    const graph = new PolyGraph()
    graph.addNode(node('stable', { value: 1 }))
    const snapshot = await graph.snapshot()

    graph.updateNode('stable', { value: 2 })
    graph.addNode(node('later', { value: 3 }))

    expect(snapshot.getNode('stable')?.data.value).toBe(1)
    expect(snapshot.getNode('later')).toBeUndefined()
    expect(snapshot.query().where('value', 1).ids()).toEqual(['stable'])
  })
})

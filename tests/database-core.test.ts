import { describe, expect, it } from 'vitest'
import { AdapterCapabilityError, ConflictError, MemoryAdapter, PolyGraph } from '../src/index'
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

  it('supports conditional replacement through addNode and transactions', async () => {
    const graph = new PolyGraph()
    graph.addNode(node('n', { value: 1 }))
    expect(() => graph.addNode(node('n', { value: 2 }), { expectedRevision: 0 })).not.toThrow()
    expect(graph.getNode('n')?.data.value).toBe(2)
    await expect(graph.transaction(tx => tx.addNode(node('n', { value: 3 }), { expectedRevision: 0 })))
      .rejects.toThrow(ConflictError)
    expect(graph.getNode('n')?.data.value).toBe(2)
  })

  it('preserves node revisions through persisted queries and warm reloads', async () => {
    const adapter = new MemoryAdapter()
    const graph = new PolyGraph(adapter)
    graph.addNode(node('n'))
    graph.updateNode('n', { value: 2 }, { expectedRevision: 0 })
    await graph.flush()
    expect((await graph.queryPersisted().first())?.revision).toBe(1)

    const reloaded = new PolyGraph(adapter)
    await reloaded.warm()
    expect(reloaded.getNode('n')?.revision).toBe(1)
    expect(() => reloaded.updateNode('n', { value: 3 }, { expectedRevision: 0 })).toThrow(ConflictError)
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

  it('rejects stale conditional edge additions', async () => {
    const graph = new PolyGraph()
    graph.addNode(node('a'))
    graph.addNode(node('b'))
    const edge = { id: 'claim', source: 'a', target: 'b', type: 'CLAIMS', createdAt: 1, revision: 3 }
    graph.addEdge(edge, { expectedRevision: 0 })
    expect(() => graph.addEdge(edge, { expectedRevision: 0 })).toThrow(ConflictError)
    await expect(graph.transaction(tx => tx.addEdge(edge, { expectedRevision: 0 }))).rejects.toThrow(ConflictError)
  })

  it('updates edge data with revisions and rejects stale edge updates', async () => {
    const graph = new PolyGraph()
    graph.addNode(node('a'))
    graph.addNode(node('b'))
    graph.addEdge({ id: 'claim', source: 'a', target: 'b', type: 'CLAIMS', data: { source: 'archive' }, createdAt: 1 })
    const updated = graph.updateEdge('claim', { confidence: 0.9 }, { expectedRevision: 0 })
    expect(updated?.data).toEqual({ source: 'archive', confidence: 0.9 })
    expect(updated?.revision).toBe(1)
    expect(() => graph.updateEdge('claim', { confidence: 0.1 }, { expectedRevision: 0 })).toThrow(ConflictError)
    await expect(graph.transaction(tx => tx.updateEdge('claim', { reviewed: true }, { expectedRevision: 1 }))).resolves.toBeDefined()
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

  it('checkpoints, backs up, restores, and verifies a binary store', async () => {
    const sourceIo = new MemoryFileIO()
    const source = new BinaryStoreAdapter({ storeDir: 'admin-source', fileIO: sourceIo })
    const graph = new PolyGraph(source)
    graph.addNode(node('a'))
    graph.addNode(node('b'))
    graph.addEdge('a', 'LINKS', 'b')
    await graph.flush()
    await source.checkpoint()
    expect((await source.verify()).ok).toBe(true)

    const backupIo = new MemoryFileIO()
    await source.backup(backupIo)
    const restoreIo = new MemoryFileIO()
    await BinaryStoreAdapter.restore(backupIo, restoreIo)
    const restored = new BinaryStoreAdapter({ storeDir: 'admin-restored', fileIO: restoreIo })
    await restored.getNode('a')
    expect((await restored.verify()).edgeCount).toBe(1)
    expect((await restored.getMutationsSince!(0n)).length).toBeGreaterThan(0)
  })

  it('reports graph and storage statistics', async () => {
    const graph = new PolyGraph(new MemoryAdapter())
    graph.defineIndex({ name: 'value', fields: ['value'] })
    graph.addNode(node('a', { value: 1 }))
    const before = await graph.stats()
    expect(before.loadedNodeCount).toBe(1)
    expect(before.persistedNodeCount).toBe(0)
    expect(before.dirtyRecordCount).toBeGreaterThan(0)
    await graph.flush()
    const after = await graph.stats()
    expect(after.persistedNodeCount).toBe(1)
    expect(after.indexCount).toBe(1)
    expect(after.pendingPersistence).toBe(false)
    await graph.queryPersisted().where('value', 1).ids()
    const observed = await graph.stats()
    expect(observed.queryCount).toBe(1)
    expect(observed.queryScannedRecords).toBeGreaterThan(0)
    expect(observed.queryIndexUsage).toEqual({ value: 1 })
  })

  it('reports and enforces adapter capability declarations', async () => {
    const graph = new PolyGraph(new MemoryAdapter())
    expect(graph.adapterCapabilities?.atomicBatches).toBe(true)
    expect(() => graph.requireAdapterCapabilities({ transactions: true })).not.toThrow()
    const unsafe = new MemoryAdapter()
    Object.defineProperty(unsafe, 'capabilities', { value: { ...unsafe.capabilities, transactions: false } })
    await expect(new PolyGraph(unsafe).transaction(() => undefined)).rejects.toBeInstanceOf(AdapterCapabilityError)
  })
})

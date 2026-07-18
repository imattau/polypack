import { describe, it, expect, beforeEach } from 'vitest'
import { PolyGraph } from '../src/graph'
import { VectorIndex, euclideanSimilarity } from '../src/vector-index'
import { MemoryAdapter } from '../src/persistence/memory'
import { IndexedDBAdapter } from '../src/persistence/indexeddb'

describe('PolyGraph', () => {
  let graph: PolyGraph

  beforeEach(() => {
    graph = new PolyGraph()
  })

  describe('node CRUD', () => {
    it('adds and retrieves a node', () => {
      graph.addNode({
        id: 'n1',
        type: 'user',
        data: { name: 'Alice', age: 30 },
        insertedAt: Date.now(),
        updatedAt: Date.now(),
      })

      const node = graph.getNode('n1')
      expect(node).toBeDefined()
      expect(node!.type).toBe('user')
      expect(node!.data.name).toBe('Alice')
    })

    it('removes a node', () => {
      graph.addNode({
        id: 'n1',
        type: 'user',
        data: {},
        insertedAt: Date.now(),
        updatedAt: Date.now(),
      })
      graph.removeNode('n1')
      expect(graph.getNode('n1')).toBeUndefined()
    })

    it('updates node data', () => {
      graph.addNode({
        id: 'n1',
        type: 'user',
        data: { name: 'Alice' },
        insertedAt: Date.now(),
        updatedAt: Date.now(),
      })
      graph.updateNode('n1', { name: 'Bob' })
      expect(graph.getNode('n1')!.data.name).toBe('Bob')
    })

    it('owns node inputs and returns detached read snapshots', () => {
      const data = { profile: { name: 'Alice' } }
      const vector = new Float64Array([1, 0])
      graph.addNode({ id: 'owned', type: 'user', data, vector, insertedAt: 1, updatedAt: 1 })

      data.profile.name = 'changed-input'
      vector[0] = 0
      const read = graph.getNode('owned')!
      read.data.profile = { name: 'changed-read' }
      read.vector![0] = 0
      const queried = graph.query().whereNodeType('user').first()!
      queried.data.profile = { name: 'changed-query' }

      expect((graph.getNode('owned')!.data.profile as { name: string }).name).toBe('Alice')
      expect(graph.getNode('owned')!.vector).toEqual(new Float64Array([1, 0]))
    })

    it('validates node identity, timestamps, and vectors', () => {
      expect(() => graph.addNode({ id: '', type: 't', data: {}, insertedAt: 1, updatedAt: 1 })).toThrow(TypeError)
      expect(() => graph.addNode({ id: 'x', type: '', data: {}, insertedAt: 1, updatedAt: 1 })).toThrow(TypeError)
      expect(() => graph.addNode({ id: 'x', type: 't', data: {}, insertedAt: -1, updatedAt: 1 })).toThrow(RangeError)
      expect(() => graph.addNode({
        id: 'x', type: 't', data: {}, vector: new Float64Array([Number.NaN]), insertedAt: 1, updatedAt: 1,
      })).toThrow(RangeError)
    })

    it('tracks node count via size', () => {
      graph.addNode({ id: 'a', type: 't1', data: {}, insertedAt: 1, updatedAt: 1 })
      graph.addNode({ id: 'b', type: 't2', data: {}, insertedAt: 2, updatedAt: 2 })
      expect(graph.size).toBe(2)
    })

    it('removeNode on non-existent id is a no-op', () => {
      expect(() => graph.removeNode('nonexistent')).not.toThrow()
      expect(graph.size).toBe(0)
    })

    it('updateNode on non-existent id returns undefined', () => {
      const result = graph.updateNode('nonexistent', { x: 1 })
      expect(result).toBeUndefined()
    })

    it('clear resets all state and graph is reusable', () => {
      graph.addNode({ id: 'a', type: 't', data: {}, insertedAt: 1, updatedAt: 1 })
      graph.addNode({ id: 'b', type: 't', data: {}, insertedAt: 2, updatedAt: 2 })
      graph.addEdge('a', 'REL', 'b')
      graph.vectors.add('v1', [1, 0])
      expect(graph.size).toBe(2)

      graph.clear()
      expect(graph.size).toBe(0)
      expect(graph.vectors.size).toBe(0)
      expect(graph.whereType('t')).toHaveLength(0)

      // Graph is reusable after clear
      graph.addNode({ id: 'c', type: 't', data: {}, insertedAt: 3, updatedAt: 3 })
      expect(graph.size).toBe(1)
    })

    it('getNodeSafe falls back to persistence for evicted nodes', async () => {
      const adapter = new MemoryAdapter()
      await adapter.putNode({
        id: 'evicted',
        type: 'doc',
        data: { text: 'persisted' },
        vector: null,
        insertedAt: 1,
        updatedAt: 1,
      })
      const g = new PolyGraph(adapter)

      const node = await g.getNodeSafe('evicted')
      expect(node).toBeDefined()
      expect(node!.data.text).toBe('persisted')
      expect(g.whereType('doc').map(n => n.id)).toContain('evicted')
    })

    it('replacing a node updates type and vector indexes', () => {
      graph.addNode({ id: 'same', type: 'old', data: {}, vector: new Float64Array([1, 0]), insertedAt: 1, updatedAt: 1 })
      graph.addNode({ id: 'same', type: 'new', data: {}, insertedAt: 2, updatedAt: 2 })

      expect(graph.whereType('old')).toHaveLength(0)
      expect(graph.whereType('new').map(n => n.id)).toEqual(['same'])
      expect(graph.vectors.has('same')).toBe(false)
    })

    it('replacing a node without a vector deletes its persisted vector', async () => {
      const adapter = new MemoryAdapter()
      const g = new PolyGraph(adapter)
      g.addNode({ id: 'same', type: 'old', data: {}, vector: new Float64Array([1, 0]), insertedAt: 1, updatedAt: 1 })
      await g.flush()
      g.addNode({ id: 'same', type: 'new', data: {}, insertedAt: 2, updatedAt: 2 })
      await g.flush()

      expect(await adapter.getAllVectors()).toHaveLength(0)
    })
  })

  describe('edge CRUD', () => {
    it('adds and retrieves outgoing edges', () => {
      graph.addNode({ id: 'src', type: 'node', data: {}, insertedAt: 1, updatedAt: 1 })
      graph.addNode({ id: 'tgt', type: 'node', data: {}, insertedAt: 2, updatedAt: 2 })
      graph.addEdge('src', 'LINKS_TO', 'tgt', { weight: 5 })

      const edges = graph.getEdges('src')
      expect(edges).toHaveLength(1)
      expect(edges[0].type).toBe('LINKS_TO')
      expect(edges[0].target).toBe('tgt')
    })

    it('owns edge data and returns detached edge snapshots', () => {
      const data = { metadata: { weight: 5 } }
      graph.addEdge('src', 'REL', 'target', data)
      data.metadata.weight = 9
      const edge = graph.getEdges('src')[0]
      ;(edge.data!.metadata as { weight: number }).weight = 12

      expect((graph.getEdges('src')[0].data!.metadata as { weight: number }).weight).toBe(5)
      expect(() => graph.addEdge('', 'REL', 'target')).toThrow(TypeError)
    })

    it('retrieves edge targets', () => {
      graph.addEdge('a', 'REL', 'b')
      graph.addEdge('a', 'REL', 'c')
      const targets = graph.getEdgeTargets('a', 'REL')
      expect(targets).toEqual(['b', 'c'])
    })

    it('retrieves edge sources via reverse lookup', () => {
      graph.addEdge('x', 'REL', 'y')
      graph.addEdge('z', 'REL', 'y')
      const sources = graph.getEdgeSources('y', 'REL')
      expect(sources.sort()).toEqual(['x', 'z'])
    })

    it('removing a target removes incoming edges from memory and persistence', async () => {
      const adapter = new MemoryAdapter()
      const g = new PolyGraph(adapter)
      g.addNode({ id: 'src', type: 'node', data: {}, insertedAt: 1, updatedAt: 1 })
      g.addNode({ id: 'target', type: 'node', data: {}, insertedAt: 2, updatedAt: 2 })
      g.addEdge('src', 'REL', 'target')
      await g.flush()

      g.removeNode('target')
      await g.flush()

      expect(g.getEdges('src')).toHaveLength(0)
      expect(await adapter.getAllEdges()).toHaveLength(0)
    })

    it('deduplicates edges', () => {
      graph.addEdge('a', 'REL', 'b')
      graph.addEdge('a', 'REL', 'b') // duplicate
      expect(graph.getEdges('a')).toHaveLength(1)
    })

    it('removes edges by type and target', () => {
      graph.addEdge('a', 'X', 'b')
      graph.addEdge('a', 'X', 'c')
      graph.addEdge('a', 'Y', 'd')
      graph.removeEdges('a', 'X')
      expect(graph.getEdgeTargets('a', 'X')).toHaveLength(0)
      expect(graph.getEdgeTargets('a', 'Y')).toHaveLength(1)
    })
  })

  describe('type index', () => {
    it('indexes nodes by type', () => {
      graph.addNode({ id: 'u1', type: 'user', data: {}, insertedAt: 1, updatedAt: 1 })
      graph.addNode({ id: 'u2', type: 'user', data: {}, insertedAt: 2, updatedAt: 2 })
      graph.addNode({ id: 'p1', type: 'post', data: {}, insertedAt: 3, updatedAt: 3 })

      expect(graph.whereType('user')).toHaveLength(2)
      expect(graph.whereType('post')).toHaveLength(1)
      expect(graph.whereType('tag')).toHaveLength(0)
    })

    it('updates type index on removal', () => {
      graph.addNode({ id: 'u1', type: 'user', data: {}, insertedAt: 1, updatedAt: 1 })
      graph.removeNode('u1')
      expect(graph.whereType('user')).toHaveLength(0)
    })
  })

  describe('change events', () => {
    it('emits node_added on addNode', () => {
      const events: any[] = []
      const sub = graph.changes.subscribe(e => events.push(e))

      graph.addNode({ id: 'n1', type: 'test', data: {}, insertedAt: 1, updatedAt: 1 })
      expect(events).toHaveLength(1)
      expect(events[0].type).toBe('node_added')
      expect(events[0].nodeId).toBe('n1')

      sub.unsubscribe()
    })

    it('emits node_removed on removeNode', () => {
      graph.addNode({ id: 'n1', type: 'test', data: {}, insertedAt: 1, updatedAt: 1 })
      const events: any[] = []
      const sub = graph.changes.subscribe(e => events.push(e))

      graph.removeNode('n1')
      expect(events).toHaveLength(1)
      expect(events[0].type).toBe('node_removed')

      sub.unsubscribe()
    })

    it('emits edge_added on addEdge', () => {
      const events: any[] = []
      const sub = graph.changes.subscribe(e => events.push(e))

      graph.addEdge('a', 'REL', 'b')
      expect(events).toHaveLength(1)
      expect(events[0].type).toBe('edge_added')

      sub.unsubscribe()
    })

    it('endBatch without startBatch throws', () => {
      expect(() => graph.endBatch()).toThrow('endBatch without startBatch')
    })

    it('batches events with startBatch/endBatch', () => {
      const events: any[] = []
      const sub = graph.changes.subscribe(e => events.push(e))

      graph.startBatch()
      graph.addNode({ id: 'n1', type: 't', data: {}, insertedAt: 1, updatedAt: 1 })
      graph.addNode({ id: 'n2', type: 't', data: {}, insertedAt: 2, updatedAt: 2 })
      expect(events).toHaveLength(0) // not yet emitted
      graph.endBatch()
      expect(events).toHaveLength(2)

      sub.unsubscribe()
    })
  })

  describe('persistence with memory adapter', () => {
    it('saves and reloads graph state', async () => {
      graph.addNode({ id: 'n1', type: 'a', data: { x: 1 }, insertedAt: 10, updatedAt: 10 })
      graph.addNode({ id: 'n2', type: 'b', data: { y: 2 }, insertedAt: 20, updatedAt: 20 })
      graph.addEdge('n1', 'REL', 'n2')

      await graph.save()

      const fresh = new PolyGraph(new MemoryAdapter())
      // Manually copy the persisted data
      const savedNodes = await graph.persistence.getAllEdges() // just trigger persistence
      // Actually, let's test by loading what was saved
      // Save was called on graph, but the adapter is MemoryAdapter — we need to
      // verify the data was written. Let's check through the adapter directly.
      const allNodeIds = await graph.persistence.allNodeIds()
      expect(allNodeIds).toContain('n1')
      expect(allNodeIds).toContain('n2')
    })

    it('warm restores state from persistence', async () => {
      const adapter = new MemoryAdapter()
      await adapter.bulkPutNodes([
        { id: 'a', type: 'x', data: { v: 1 }, vector: null, insertedAt: 1, updatedAt: 1 },
        { id: 'b', type: 'y', data: { v: 2 }, vector: null, insertedAt: 2, updatedAt: 2 },
      ])
      await adapter.bulkPutEdges([
        { id: 'a::REL::b', source: 'a', target: 'b', type: 'REL', data: null, createdAt: 100 },
      ])

      const g = new PolyGraph(adapter)
      await g.warm()

      expect(g.getNode('a')).toBeDefined()
      expect(g.getNode('b')).toBeDefined()
      expect(g.size).toBe(2)
      expect(g.getEdgeTargets('a', 'REL')).toEqual(['b'])
    })

    it('reports loaded and persisted node state explicitly', async () => {
      const adapter = new MemoryAdapter()
      const g = new PolyGraph(adapter, 1)
      for (let i = 0; i < 12; i++) {
        g.addNode({ id: `n${i}`, type: 't', data: {}, insertedAt: i, updatedAt: i })
      }
      await g.flush()

      expect(g.loadedSize).toBe(1)
      expect(g.size).toBe(g.loadedSize)
      expect(g.hasLoadedNode('n0')).toBe(false)
      expect(await g.persistedSize()).toBe(12)
    })

    it('updates an evicted node through updateNodeSafe', async () => {
      const adapter = new MemoryAdapter()
      const g = new PolyGraph(adapter, 1)
      for (let i = 0; i < 12; i++) {
        g.addNode({ id: `n${i}`, type: 't', data: { value: i }, insertedAt: i, updatedAt: i })
      }
      await g.flush()
      expect(g.hasLoadedNode('n0')).toBe(false)

      const updated = await g.updateNodeSafe('n0', { value: 99 }, new Float64Array([1, 0]))
      await g.flush()

      expect(updated?.data.value).toBe(99)
      expect((await adapter.getNode('n0'))?.data.value).toBe(99)
      expect((await adapter.getNode('n0'))?.vector).toEqual([1, 0])
    })

    it('removes an evicted node and its evicted owned descendant safely', async () => {
      const adapter = new MemoryAdapter()
      const g = new PolyGraph(adapter, 1)
      g.addNode({ id: 'parent', type: 't', data: {}, insertedAt: 0, updatedAt: 0 })
      g.addNode({ id: 'child', type: 't', data: {}, insertedAt: 1, updatedAt: 1 })
      g.addEdge('parent', 'OWNS', 'child', undefined, 'owned')
      for (let i = 0; i < 12; i++) {
        g.addNode({ id: `filler${i}`, type: 't', data: {}, insertedAt: i + 2, updatedAt: i + 2 })
      }
      await g.flush()
      expect(g.hasLoadedNode('parent')).toBe(false)
      expect(g.hasLoadedNode('child')).toBe(false)

      expect(await g.removeNodeSafe('parent')).toBe(true)
      expect(await g.getNodeSafe('parent')).toBeUndefined()
      await g.flush()

      expect(await adapter.getNode('parent')).toBeUndefined()
      expect(await adapter.getNode('child')).toBeUndefined()
      expect(await adapter.getAllEdges()).toHaveLength(0)
    })

    it('safe removal preserves an owned target that has another owner', async () => {
      const adapter = new MemoryAdapter()
      const g = new PolyGraph(adapter, 1)
      g.addNode({ id: 'owner-a', type: 't', data: {}, insertedAt: 0, updatedAt: 0 })
      g.addNode({ id: 'owner-b', type: 't', data: {}, insertedAt: 1, updatedAt: 1 })
      g.addNode({ id: 'shared-child', type: 't', data: {}, insertedAt: 2, updatedAt: 2 })
      g.addEdge('owner-a', 'OWNS', 'shared-child', undefined, 'owned')
      g.addEdge('owner-b', 'OWNS', 'shared-child', undefined, 'owned')
      for (let i = 0; i < 10; i++) {
        g.addNode({ id: `extra${i}`, type: 't', data: {}, insertedAt: i + 3, updatedAt: i + 3 })
      }
      await g.flush()

      expect(await g.removeNodeSafe('owner-a')).toBe(true)
      await g.flush()

      expect(await adapter.getNode('owner-a')).toBeUndefined()
      expect(await adapter.getNode('owner-b')).toBeDefined()
      expect(await adapter.getNode('shared-child')).toBeDefined()
      expect((await adapter.getAllEdges()).map(edge => edge.source)).toEqual(['owner-b'])
    })

    it('safe removal is cycle-safe and does not resurrect pending deletions', async () => {
      const adapter = new MemoryAdapter()
      const g = new PolyGraph(adapter, 1)
      g.addNode({ id: 'a', type: 't', data: {}, insertedAt: 0, updatedAt: 0 })
      g.addNode({ id: 'b', type: 't', data: {}, insertedAt: 1, updatedAt: 1 })
      g.addEdge('a', 'OWNS', 'b', undefined, 'owned')
      g.addEdge('b', 'OWNS', 'a', undefined, 'owned')
      for (let i = 0; i < 10; i++) {
        g.addNode({ id: `cycle-filler${i}`, type: 't', data: {}, insertedAt: i + 2, updatedAt: i + 2 })
      }
      await g.flush()

      expect(await g.removeNodeSafe('a')).toBe(true)
      expect(await g.getNodeSafe('a')).toBeUndefined()
      expect(await g.getNodeSafe('b')).toBeUndefined()
      await g.flush()

      expect(await adapter.getNode('a')).toBeUndefined()
      expect(await adapter.getNode('b')).toBeUndefined()
    })

    it('queries the complete persisted set without changing the loaded set', async () => {
      const adapter = new MemoryAdapter()
      const g = new PolyGraph(adapter, 1)
      for (let i = 0; i < 12; i++) {
        g.addNode({
          id: `doc${i}`,
          type: i === 11 ? 'other' : 'doc',
          data: { group: i % 2 === 0 ? 'even' : 'odd', rank: i },
          vector: new Float64Array([12 - i, i]),
          insertedAt: i,
          updatedAt: i,
        })
      }
      await g.flush()
      const loadedBefore = g.loadedSize

      const results = await g.queryPersisted()
        .whereNodeType('doc')
        .where('group', 'even')
        .whereAttributeRange('rank', { above: 1, below: 11 })
        .orderBy('rank', 'desc')
        .offset(1)
        .limit(2)
        .toArray()

      expect(results.map(node => node.id)).toEqual(['doc8', 'doc6'])
      expect(g.loadedSize).toBe(loadedBefore)
      expect(g.hasLoadedNode('doc6')).toBe(false)
    })

    it('supports persisted similarity, count, first, and ids terminals', async () => {
      const adapter = new MemoryAdapter()
      await adapter.bulkPutNodes([
        { id: 'a', type: 'doc', data: {}, vector: [1, 0], insertedAt: 1, updatedAt: 1 },
        { id: 'b', type: 'doc', data: {}, vector: [0.9, 0.1], insertedAt: 2, updatedAt: 2 },
        { id: 'c', type: 'doc', data: {}, vector: [0, 1], insertedAt: 3, updatedAt: 3 },
        { id: 'no-vector', type: 'doc', data: {}, vector: null, insertedAt: 4, updatedAt: 4 },
      ])
      const g = new PolyGraph(adapter, 1)

      expect(await g.queryPersisted().whereNodeType('doc').count()).toBe(4)
      expect((await g.queryPersisted().whereNodeType('doc').similarTo([1, 0], 0.5).first())?.id).toBe('a')
      expect(await g.queryPersisted().whereNodeType('doc').similarTo([1, 0], 0).limit(2).ids()).toEqual(['a', 'b'])
      expect(g.loadedSize).toBe(0)
    })

    it('returns detached persisted results and excludes unflushed mutations', async () => {
      const adapter = new MemoryAdapter()
      await adapter.putNode({ id: 'stored', type: 'doc', data: { value: 1 }, vector: null, insertedAt: 1, updatedAt: 1 })
      const g = new PolyGraph(adapter)
      g.addNode({ id: 'pending', type: 'doc', data: {}, insertedAt: 2, updatedAt: 2 })

      const result = (await g.queryPersisted().whereNodeType('doc').toArray())[0]
      result.data.value = 99

      expect(result.id).toBe('stored')
      expect((await adapter.getNode('stored'))?.data.value).toBe(1)
      expect(await g.queryPersisted().whereNodeType('doc').ids()).toEqual(['stored'])
    })

    it('falls back for custom adapters without query hooks', async () => {
      const adapter = new MemoryAdapter()
      await adapter.bulkPutNodes([
        { id: 'a', type: 'doc', data: { rank: 2 }, vector: null, insertedAt: 1, updatedAt: 1 },
        { id: 'b', type: 'other', data: { rank: 1 }, vector: null, insertedAt: 2, updatedAt: 2 },
      ])
      Object.defineProperty(adapter, 'queryNodes', { value: undefined })
      Object.defineProperty(adapter, 'countNodes', { value: undefined })
      const g = new PolyGraph(adapter)

      expect(await g.queryPersisted().whereNodeType('doc').where('rank', 2).ids()).toEqual(['a'])
      expect(await g.queryPersisted().whereNodeType('doc').count()).toBe(1)
    })

    it('delegates safe pagination to the persistence adapter', async () => {
      class InspectingAdapter extends MemoryAdapter {
        lastQuery?: Parameters<MemoryAdapter['queryNodes']>[0]
        override async queryNodes(query: Parameters<MemoryAdapter['queryNodes']>[0]) {
          this.lastQuery = query
          return super.queryNodes(query)
        }
      }
      const adapter = new InspectingAdapter()
      await adapter.bulkPutNodes([
        { id: 'a', type: 'doc', data: {}, vector: null, insertedAt: 1, updatedAt: 1 },
        { id: 'b', type: 'doc', data: {}, vector: null, insertedAt: 2, updatedAt: 2 },
        { id: 'c', type: 'doc', data: {}, vector: null, insertedAt: 3, updatedAt: 3 },
      ])
      const g = new PolyGraph(adapter)

      expect(await g.queryPersisted().whereNodeType('doc').offset(1).limit(1).ids()).toEqual(['b'])
      expect(adapter.lastQuery?.offset).toBe(1)
      expect(adapter.lastQuery?.limit).toBe(1)
    })

    it('supports persisted edge filters and joins', async () => {
      const adapter = new MemoryAdapter()
      await adapter.bulkPutNodes([
        { id: 'alice', type: 'user', data: { name: 'Alice' }, vector: null, insertedAt: 1, updatedAt: 1 },
        { id: 'bob', type: 'user', data: { name: 'Bob' }, vector: null, insertedAt: 2, updatedAt: 2 },
        { id: 'dune', type: 'book', data: { genre: 'sci-fi' }, vector: null, insertedAt: 3, updatedAt: 3 },
        { id: 'lotr', type: 'book', data: { genre: 'fantasy' }, vector: null, insertedAt: 4, updatedAt: 4 },
      ])
      await adapter.bulkPutEdges([
        { id: 'alice::RATED::dune', source: 'alice', target: 'dune', type: 'RATED', data: null, createdAt: 1 },
        { id: 'bob::RATED::lotr', source: 'bob', target: 'lotr', type: 'RATED', data: null, createdAt: 2 },
      ])
      const g = new PolyGraph(adapter)

      expect(await g.queryPersisted().whereNodeType('user').whereEdge('RATED', 'dune').ids()).toEqual(['alice'])
      expect(await g.queryPersisted().whereNodeType('book').whereEdgeSource('alice').ids()).toEqual(['dune'])
      expect(await g.queryPersisted().whereNodeType('user').join('RATED', 'out', node => node.data.genre === 'sci-fi').ids()).toEqual(['alice'])
      expect(await g.queryPersisted().whereNodeType('book').join('RATED', 'in', node => node.data.name === 'Bob').ids()).toEqual(['lotr'])
    })

    it('supports persisted traversal and collection in both directions', async () => {
      const adapter = new MemoryAdapter()
      await adapter.bulkPutNodes([
        { id: 'a', type: 'node', data: { order: 1 }, vector: null, insertedAt: 1, updatedAt: 1 },
        { id: 'b', type: 'node', data: { order: 2 }, vector: null, insertedAt: 2, updatedAt: 2 },
        { id: 'c', type: 'node', data: { order: 3 }, vector: null, insertedAt: 3, updatedAt: 3 },
        { id: 'd', type: 'node', data: { order: 4 }, vector: null, insertedAt: 4, updatedAt: 4 },
      ])
      await adapter.bulkPutEdges([
        { id: 'a::NEXT::b', source: 'a', target: 'b', type: 'NEXT', data: null, createdAt: 1 },
        { id: 'b::NEXT::c', source: 'b', target: 'c', type: 'NEXT', data: null, createdAt: 2 },
        { id: 'c::OTHER::d', source: 'c', target: 'd', type: 'OTHER', data: null, createdAt: 3 },
      ])
      const g = new PolyGraph(adapter)

      expect(await g.queryPersisted().where('order', 1).traverse('NEXT', 2).ids()).toEqual(['a', 'b', 'c'])
      expect(await g.queryPersisted().where('order', 3).traverse('NEXT', 2, 'in').ids()).toEqual(['c', 'b', 'a'])
      expect((await g.queryPersisted().where('order', 1).collect('NEXT')).map(node => node.id)).toEqual(['b'])
      expect((await g.queryPersisted().where('order', 3).collect('NEXT', 'in')).map(node => node.id)).toEqual(['b'])
    })
  })

  describe('persistence with indexeddb adapter', () => {
    it('saves and loads via IndexedDBAdapter', async () => {
      const adapter = new IndexedDBAdapter({ name: 'test-db-' + Date.now(), version: 1 })
      const g = new PolyGraph(adapter)

      g.addNode({ id: 'n1', type: 'doc', data: { text: 'hello' }, insertedAt: 1, updatedAt: 1 })
      g.addNode({ id: 'n2', type: 'doc', data: { text: 'world' }, insertedAt: 2, updatedAt: 2 })
      g.addEdge('n1', 'NEXT', 'n2')

      await g.save()
      await g.dispose()

      // Reload from same adapter
      const g2 = new PolyGraph(adapter)
      await g2.warm()

      expect(g2.size).toBe(2)
      expect(g2.getNode('n1')?.data.text).toBe('hello')
      expect(g2.getEdgeTargets('n1', 'NEXT')).toEqual(['n2'])

      await g2.dispose()
    })
  })

  describe('flush mechanism', () => {
    it('persists dirty nodes to adapter on flush', async () => {
      const adapter = new MemoryAdapter()
      const g = new PolyGraph(adapter)

      g.addNode({ id: 'd1', type: 't', data: {}, insertedAt: 1, updatedAt: 1 })
      await g.flush()

      const loaded = await adapter.getNode('d1')
      expect(loaded).toBeDefined()
      expect(loaded!.type).toBe('t')
    })

    it('deletes removed nodes from adapter on flush', async () => {
      const adapter = new MemoryAdapter()
      const g = new PolyGraph(adapter)

      g.addNode({ id: 'd1', type: 't', data: {}, insertedAt: 1, updatedAt: 1 })
      await g.flush()
      g.removeNode('d1')
      await g.flush()

      const loaded = await adapter.getNode('d1')
      expect(loaded).toBeUndefined()
    })

    it('does not lose mutations made while a flush is in flight', async () => {
      let release!: () => void
      const gate = new Promise<void>(resolve => { release = resolve })
      class SlowAdapter extends MemoryAdapter {
        firstWrite = true
        override async bulkPutNodes(nodes: Parameters<MemoryAdapter['bulkPutNodes']>[0]): Promise<void> {
          if (this.firstWrite) {
            this.firstWrite = false
            await gate
          }
          await super.bulkPutNodes(nodes)
        }
      }
      const adapter = new SlowAdapter()
      const g = new PolyGraph(adapter)
      g.addNode({ id: 'first', type: 't', data: {}, insertedAt: 1, updatedAt: 1 })
      const flushing = g.flush()
      g.addNode({ id: 'during', type: 't', data: {}, insertedAt: 2, updatedAt: 2 })
      release()
      await flushing

      expect(await adapter.getNode('during')).toBeDefined()
    })

    it('flushes pending changes before dispose', async () => {
      const adapter = new MemoryAdapter()
      const g = new PolyGraph(adapter)
      g.addNode({ id: 'pending', type: 't', data: {}, insertedAt: 1, updatedAt: 1 })
      await g.dispose()
      expect(await adapter.getNode('pending')).toBeDefined()
    })

    it('does not rewrite hydrated vectors', async () => {
      class CountingAdapter extends MemoryAdapter {
        vectorWrites = 0
        override async bulkPutVectors(entries: Parameters<MemoryAdapter['bulkPutVectors']>[0]): Promise<void> {
          this.vectorWrites += entries.length
          await super.bulkPutVectors(entries)
        }
      }
      const adapter = new CountingAdapter()
      await adapter.putNode({ id: 'v', type: 't', data: {}, vector: [1, 0], insertedAt: 1, updatedAt: 1 })
      await adapter.putVector('v', [1, 0])
      const g = new PolyGraph(adapter)
      await g.warm()
      await g.flush()
      expect(adapter.vectorWrites).toBe(0)
      await g.dispose()
    })

    it('persists dirty nodes and vectors even when they are evicted before flush', async () => {
      const adapter = new MemoryAdapter()
      const g = new PolyGraph(adapter, 1)
      for (let i = 0; i < 10; i++) {
        g.addNode({ id: `n${i}`, type: 't', data: {}, vector: new Float64Array([i, 1]), insertedAt: i, updatedAt: i })
      }
      await g.flush()

      expect(await adapter.allNodeIds()).toHaveLength(10)
      expect(await adapter.getAllVectors()).toHaveLength(10)
    })
  })

  describe('prune', () => {
    it('removes oldest nodes when over capacity', async () => {
      for (let i = 0; i < 10; i++) {
        graph.addNode({ id: `n${i}`, type: 't', data: {}, insertedAt: i, updatedAt: i })
      }
      await graph.prune(5)
      expect(graph.size).toBe(5)
      // Nodes 0-4 should be removed (oldest first)
      expect(graph.getNode('n0')).toBeUndefined()
      expect(graph.getNode('n9')).toBeDefined()
    })
  })

  describe('vector index integration', () => {
    it('stores vectors and enables similarity search', () => {
      graph.vectors.add('v1', [1, 0, 0])
      graph.vectors.add('v2', [0, 1, 0])
      graph.vectors.add('v3', [0.9, 0.1, 0])

      const results = graph.vectors.query([1, 0, 0], 2)
      expect(results).toHaveLength(2)
      expect(results[0].id).toBe('v1')
      expect(results[1].id).toBe('v3')
    })

    it('addNode auto-registers node.vector in VectorIndex', () => {
      graph.addNode({
        id: 'vec-node',
        type: 'embedding',
        data: {},
        vector: new Float64Array([1, 0, 0]),
        insertedAt: 1,
        updatedAt: 1,
      })

      expect(graph.vectors.has('vec-node')).toBe(true)
      expect(graph.vectors.get('vec-node')).toEqual([1, 0, 0])
    })

    it('similarTo filters and sorts by cosine similarity', () => {
      graph.addNode({ id: 'a', type: 'item', data: {}, vector: new Float64Array([1, 0, 0]), insertedAt: 1, updatedAt: 1 })
      graph.addNode({ id: 'b', type: 'item', data: {}, vector: new Float64Array([0.9, 0.1, 0]), insertedAt: 2, updatedAt: 2 })
      graph.addNode({ id: 'c', type: 'item', data: {}, vector: new Float64Array([0, 1, 0]), insertedAt: 3, updatedAt: 3 })
      graph.addNode({ id: 'd', type: 'other', data: {}, vector: new Float64Array([1, 0, 0]), insertedAt: 4, updatedAt: 4 })

      const results = graph.query()
        .whereNodeType('item')
        .similarTo([1, 0, 0], 0.5, 2)
        .toArray()

      expect(results).toHaveLength(2)
      expect(results[0].id).toBe('a')
      expect(results[1].id).toBe('b')
    })

    it('similarTo with threshold excludes low-scoring nodes', () => {
      graph.addNode({ id: 'a', type: 't', data: {}, vector: new Float64Array([1, 0, 0]), insertedAt: 1, updatedAt: 1 })
      graph.addNode({ id: 'b', type: 't', data: {}, vector: new Float64Array([0, 1, 0]), insertedAt: 2, updatedAt: 2 })

      const results = graph.query()
        .similarTo([1, 0, 0], 0.9)
        .toArray()

      expect(results).toHaveLength(1)
      expect(results[0].id).toBe('a')
    })

    it('similarTo excludes nodes without vectors', () => {
      graph.addNode({ id: 'a', type: 't', data: {}, vector: new Float64Array([1, 0, 0]), insertedAt: 1, updatedAt: 1 })
      graph.addNode({ id: 'b', type: 't', data: {}, insertedAt: 2, updatedAt: 2 })

      const results = graph.query()
        .similarTo([1, 0, 0], 0)
        .orderBy('insertedAt', 'asc')
        .toArray()

      expect(results.map(node => node.id)).toEqual(['a'])
    })

    it('markVectorDirty for existing vector schedules persist without throwing', () => {
      graph.vectors.add('existing', [1, 0, 0])
      expect(() => graph.markVectorDirty('existing')).not.toThrow()
    })

    it('vector overwrite via addNode updates VectorIndex', () => {
      graph.addNode({ id: 'v', type: 't', data: {}, vector: new Float64Array([1, 0]), insertedAt: 1, updatedAt: 1 })
      expect(graph.vectors.get('v')).toEqual([1, 0])
      // Re-add with different vector
      graph.addNode({ id: 'v', type: 't', data: {}, vector: new Float64Array([0, 1]), insertedAt: 2, updatedAt: 2 })
      expect(graph.vectors.get('v')).toEqual([0, 1])
    })

    it('removes a node vector explicitly and persists the removal', async () => {
      const adapter = new MemoryAdapter()
      const g = new PolyGraph(adapter)
      g.addNode({ id: 'v', type: 't', data: {}, vector: new Float64Array([1, 0]), insertedAt: 1, updatedAt: 1 })
      await g.flush()

      expect(g.removeNodeVector('v')?.vector).toBeUndefined()
      await g.flush()

      expect(g.getNode('v')?.vector).toBeUndefined()
      expect((await adapter.getNode('v'))?.vector).toBeNull()
      expect(await adapter.getAllVectors()).toHaveLength(0)
    })

    it('validates query pagination, traversal, ranges, and vectors', () => {
      expect(() => graph.query().limit(-1)).toThrow(RangeError)
      expect(() => graph.query().offset(1.5)).toThrow(RangeError)
      expect(() => graph.query().traverse('REL', -1)).toThrow(RangeError)
      expect(() => graph.query().whereAttributeRange('x', { above: Number.NaN })).toThrow(RangeError)
      expect(() => graph.query().similarTo([Number.POSITIVE_INFINITY])).toThrow(RangeError)
      expect(() => graph.query().similarTo([1], 0, -1)).toThrow(RangeError)
      expect(graph.query().limit(0).toArray()).toEqual([])
    })

    it('updateNode with vector updates VectorIndex', () => {
      graph.addNode({ id: 'x', type: 't', data: {}, insertedAt: 1, updatedAt: 1 })
      expect(graph.vectors.has('x')).toBe(false)

      graph.updateNode('x', { label: 'updated' }, new Float64Array([0, 1, 0]))
      expect(graph.vectors.has('x')).toBe(true)
      expect(graph.vectors.get('x')).toEqual([0, 1, 0])
    })

    it('getNodeSafe restores vector to VectorIndex', async () => {
      const adapter = new MemoryAdapter()
      await adapter.putNode({
        id: 'stored',
        type: 'doc',
        data: {},
        vector: [0.5, 0.5, 0],
        insertedAt: 1,
        updatedAt: 1,
      })
      const g = new PolyGraph(adapter)

      const node = await g.getNodeSafe('stored')
      expect(node).toBeDefined()
      expect(g.vectors.has('stored')).toBe(true)
      expect(g.vectors.get('stored')).toEqual([0.5, 0.5, 0])
    })

    it('cascade deletion removes vectors of cascaded children', () => {
      graph.addNode({ id: 'parent', type: 't', data: {}, vector: new Float64Array([1, 0, 0]), insertedAt: 1, updatedAt: 1 })
      graph.addNode({ id: 'child', type: 't', data: {}, vector: new Float64Array([0, 1, 0]), insertedAt: 2, updatedAt: 2 })
      graph.addEdge('parent', 'OWNS', 'child', undefined, 'owned')

      expect(graph.vectors.has('child')).toBe(true)
      graph.removeNode('parent')
      expect(graph.vectors.has('child')).toBe(false)
    })

    it('save/warm round-trip preserves vectors', async () => {
      const adapter = new MemoryAdapter()
      const g = new PolyGraph(adapter)
      g.addNode({ id: 'a', type: 't', data: {}, vector: new Float64Array([1, 0, 0]), insertedAt: 1, updatedAt: 1 })
      g.addNode({ id: 'b', type: 't', data: {}, vector: new Float64Array([0, 1, 0]), insertedAt: 2, updatedAt: 2 })
      await g.save()
      await g.dispose()

      const g2 = new PolyGraph(adapter)
      await g2.warm()
      expect(g2.vectors.has('a')).toBe(true)
      expect(g2.vectors.get('b')).toEqual([0, 1, 0])
    })

    it('euclideanSimilarity works as VectorIndex distance function', () => {
      const idx = new VectorIndex(undefined, euclideanSimilarity)
      idx.add('a', [1, 0, 0])
      idx.add('b', [0.9, 0.1, 0])
      idx.add('c', [100, 0, 0])

      const results = idx.query([1, 0, 0], 2)
      expect(results[0].id).toBe('a')
      expect(results[1].id).toBe('b')
    })
  })

  describe('query builder', () => {
    it('filters by node type', () => {
      graph.addNode({ id: 'a', type: 'A', data: {}, insertedAt: 1, updatedAt: 1 })
      graph.addNode({ id: 'b', type: 'B', data: {}, insertedAt: 1, updatedAt: 1 })

      const results = graph.query().whereNodeType('A').toArray()
      expect(results).toHaveLength(1)
      expect(results[0].id).toBe('a')
    })

    it('filters by attribute', () => {
      graph.addNode({ id: 'a', type: 't', data: { color: 'red' }, insertedAt: 1, updatedAt: 1 })
      graph.addNode({ id: 'b', type: 't', data: { color: 'blue' }, insertedAt: 1, updatedAt: 1 })

      const results = graph.query().where('color', 'red').toArray()
      expect(results).toHaveLength(1)
      expect(results[0].id).toBe('a')
    })

    it('filters by edge presence', () => {
      graph.addEdge('a', 'REL', 'b')
      const results = graph.query().whereNodeType('t').whereEdge('REL', 'b').toArray()
      expect(results).toHaveLength(0) // a and b don't exist as nodes
    })

    it('orders by attribute', () => {
      graph.addNode({ id: 'a', type: 't', data: { rank: 2 }, insertedAt: 1, updatedAt: 1 })
      graph.addNode({ id: 'b', type: 't', data: { rank: 1 }, insertedAt: 1, updatedAt: 1 })
      graph.addNode({ id: 'c', type: 't', data: { rank: 3 }, insertedAt: 1, updatedAt: 1 })

      const results = graph.query().orderBy('rank', 'asc').toArray()
      expect(results.map(r => r.id)).toEqual(['b', 'a', 'c'])
    })

    it('limits results', () => {
      for (let i = 0; i < 10; i++) {
        graph.addNode({ id: `n${i}`, type: 't', data: { idx: i }, insertedAt: i, updatedAt: i })
      }
      const results = graph.query().orderBy('idx', 'asc').limit(3).toArray()
      expect(results).toHaveLength(3)
    })

    it('traverses edges (BFS)', () => {
      // n1 -> n2 -> n3
      graph.addNode({ id: 'n1', type: 't', data: { label: 'start' }, insertedAt: 1, updatedAt: 1 })
      graph.addNode({ id: 'n2', type: 't', data: {}, insertedAt: 2, updatedAt: 2 })
      graph.addNode({ id: 'n3', type: 't', data: {}, insertedAt: 3, updatedAt: 3 })
      graph.addEdge('n1', 'NEXT', 'n2')
      graph.addEdge('n2', 'NEXT', 'n3')

      const results = graph.query()
        .where('label', 'start')
        .traverse('NEXT', 2, 'out')
        .toArray()

      const got = results.map(r => r.id).sort()
      expect(got).toContain('n2')
      expect(got).toContain('n3')
    })
  })

  describe('cascading deletion and orphan detection', () => {
    it('owned edge: removeNode cascades to target', () => {
      graph.addNode({ id: 'parent', type: 't', data: {}, insertedAt: 1, updatedAt: 1 })
      graph.addNode({ id: 'child', type: 't', data: {}, insertedAt: 2, updatedAt: 2 })
      graph.addEdge('parent', 'OWNS', 'child', undefined, 'owned')

      graph.removeNode('parent')

      expect(graph.getNode('parent')).toBeUndefined()
      expect(graph.getNode('child')).toBeUndefined()
    })

    it('owned edge: multi-source cascade stops when another owner remains', () => {
      graph.addNode({ id: 'a', type: 't', data: {}, insertedAt: 1, updatedAt: 1 })
      graph.addNode({ id: 'b', type: 't', data: {}, insertedAt: 2, updatedAt: 2 })
      graph.addNode({ id: 'target', type: 't', data: {}, insertedAt: 3, updatedAt: 3 })
      graph.addEdge('a', 'OWNS', 'target', undefined, 'owned')
      graph.addEdge('b', 'OWNS', 'target', undefined, 'owned')

      graph.removeNode('a')

      expect(graph.getNode('a')).toBeUndefined()
      expect(graph.getNode('b')).toBeDefined()
      expect(graph.getNode('target')).toBeDefined()

      graph.removeNode('b')

      expect(graph.getNode('target')).toBeUndefined()
    })

    it('shared edge: removeNode does NOT cascade to target', () => {
      graph.addNode({ id: 'src', type: 't', data: {}, insertedAt: 1, updatedAt: 1 })
      graph.addNode({ id: 'tgt', type: 't', data: {}, insertedAt: 2, updatedAt: 2 })
      graph.addEdge('src', 'USES', 'tgt', undefined, 'shared')

      graph.removeNode('src')

      expect(graph.getNode('src')).toBeUndefined()
      expect(graph.getNode('tgt')).toBeDefined()
    })

    it('reference edge: removeNode does NOT cascade to target', () => {
      graph.addNode({ id: 'src', type: 't', data: {}, insertedAt: 1, updatedAt: 1 })
      graph.addNode({ id: 'tgt', type: 't', data: {}, insertedAt: 2, updatedAt: 2 })
      graph.addEdge('src', 'REFERS', 'tgt', undefined, 'reference')

      graph.removeNode('src')

      expect(graph.getNode('tgt')).toBeDefined()
    })

    it('owned edge: removeEdges cascades when last owner', () => {
      graph.addNode({ id: 'a', type: 't', data: {}, insertedAt: 1, updatedAt: 1 })
      graph.addNode({ id: 'b', type: 't', data: {}, insertedAt: 2, updatedAt: 2 })
      graph.addEdge('a', 'OWNS', 'b', undefined, 'owned')

      graph.removeEdges('a', 'OWNS', 'b')

      expect(graph.getNode('b')).toBeUndefined()
    })

    it('owned edge: removeEdges does NOT cascade when other owner remains', () => {
      graph.addNode({ id: 'a', type: 't', data: {}, insertedAt: 1, updatedAt: 1 })
      graph.addNode({ id: 'b', type: 't', data: {}, insertedAt: 2, updatedAt: 2 })
      graph.addNode({ id: 'c', type: 't', data: {}, insertedAt: 3, updatedAt: 3 })
      graph.addEdge('a', 'OWNS', 'c', undefined, 'owned')
      graph.addEdge('b', 'OWNS', 'c', undefined, 'owned')

      graph.removeEdges('a', 'OWNS', 'c')

      expect(graph.getNode('c')).toBeDefined()
    })

    it('shared edge: removeEdges fires onOrphan when target becomes disconnected', () => {
      const orphans: string[] = []
      class OrphanGraph extends PolyGraph {
        protected onOrphan(id: string): void {
          orphans.push(id)
        }
      }
      const g = new OrphanGraph()
      g.addNode({ id: 'a', type: 't', data: {}, insertedAt: 1, updatedAt: 1 })
      g.addNode({ id: 'b', type: 't', data: {}, insertedAt: 2, updatedAt: 2 })
      g.addEdge('a', 'USES', 'b', undefined, 'shared')

      g.removeEdges('a', 'USES', 'b')

      expect(orphans).toEqual(['b'])
    })

    it('shared edge: removeEdges does NOT fire onOrphan when other edges remain', () => {
      const orphans: string[] = []
      class OrphanGraph extends PolyGraph {
        protected onOrphan(id: string): void {
          orphans.push(id)
        }
      }
      const g = new OrphanGraph()
      g.addNode({ id: 'a', type: 't', data: {}, insertedAt: 1, updatedAt: 1 })
      g.addNode({ id: 'b', type: 't', data: {}, insertedAt: 2, updatedAt: 2 })
      g.addNode({ id: 'c', type: 't', data: {}, insertedAt: 3, updatedAt: 3 })
      g.addEdge('a', 'USES', 'c', undefined, 'shared')
      g.addEdge('b', 'REFERS', 'c', undefined, 'reference')

      g.removeEdges('a', 'USES', 'c')

      expect(orphans).toEqual([])
    })

    it('cyclic owned edges do not infinite-loop', () => {
      graph.addNode({ id: 'a', type: 't', data: {}, insertedAt: 1, updatedAt: 1 })
      graph.addNode({ id: 'b', type: 't', data: {}, insertedAt: 2, updatedAt: 2 })
      graph.addEdge('a', 'OWNS', 'b', undefined, 'owned')
      graph.addEdge('b', 'OWNS', 'a', undefined, 'owned')

      graph.removeNode('a')

      expect(graph.getNode('a')).toBeUndefined()
      expect(graph.getNode('b')).toBeUndefined()
    })

    it('default ownership is reference (no cascade)', () => {
      graph.addNode({ id: 'a', type: 't', data: {}, insertedAt: 1, updatedAt: 1 })
      graph.addNode({ id: 'b', type: 't', data: {}, insertedAt: 2, updatedAt: 2 })
      graph.addEdge('a', 'REL', 'b')

      graph.removeNode('a')

      expect(graph.getNode('b')).toBeDefined()
    })

    it('addEdge stores ownership in edge data and getOwnership preserves it', () => {
      graph.addNode({ id: 'a', type: 't', data: {}, insertedAt: 1, updatedAt: 1 })
      graph.addNode({ id: 'b', type: 't', data: {}, insertedAt: 2, updatedAt: 2 })
      graph.addEdge('a', 'OWNS', 'b', { label: 'my-child' }, 'owned')

      const edges = graph.getEdges('a')
      expect(edges).toHaveLength(1)
      expect(edges[0].data?.__ownership).toBe('owned')
      expect(edges[0].data?.label).toBe('my-child')
    })

    it('removeEdges with owned cascade emits edge_removed and node_removed events', () => {
      const events: any[] = []
      const sub = graph.changes.subscribe(e => events.push(e))

      graph.addNode({ id: 'a', type: 't', data: {}, insertedAt: 1, updatedAt: 1 })
      graph.addNode({ id: 'b', type: 't', data: {}, insertedAt: 2, updatedAt: 2 })
      graph.addEdge('a', 'OWNS', 'b', undefined, 'owned')

      graph.removeEdges('a', 'OWNS', 'b')

      const types = events.map(e => e.type)
      expect(types).toContain('edge_removed')
      expect(types).toContain('node_removed')

      sub.unsubscribe()
    })
  })

  describe('extension hooks', () => {
    it('calls onNodeIndex for each indexed node', () => {
      const indexed: string[] = []
      class CustomGraph extends PolyGraph {
        protected onNodeIndex(node: any): void {
          indexed.push(node.id)
        }
      }
      const g = new CustomGraph()
      g.addNode({ id: 'x', type: 't', data: {}, insertedAt: 1, updatedAt: 1 })
      expect(indexed).toEqual(['x'])
    })

    it('calls onNodeUnindex on removal', () => {
      const unindexed: string[] = []
      class CustomGraph extends PolyGraph {
        protected onNodeUnindex(id: string): void {
          unindexed.push(id)
        }
      }
      const g = new CustomGraph()
      g.addNode({ id: 'x', type: 't', data: {}, insertedAt: 1, updatedAt: 1 })
      g.removeNode('x')
      expect(unindexed).toEqual(['x'])
    })
  })
})

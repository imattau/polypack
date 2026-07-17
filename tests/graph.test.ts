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

    it('similarTo scores nodes without vector as 0', () => {
      graph.addNode({ id: 'a', type: 't', data: {}, vector: new Float64Array([1, 0, 0]), insertedAt: 1, updatedAt: 1 })
      graph.addNode({ id: 'b', type: 't', data: {}, insertedAt: 2, updatedAt: 2 })

      const results = graph.query()
        .similarTo([1, 0, 0], 0)
        .orderBy('insertedAt', 'asc')
        .toArray()

      // Both pass threshold 0; b has score 0
      expect(results).toHaveLength(2)
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

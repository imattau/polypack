import { describe, it, expect, beforeEach } from 'vitest'
import { BinaryStoreAdapter } from '../src/persistence/binary-store'
import { MemoryFileIO } from '../src/persistence/binary-file-io'
import type { SerializedNode, SerializedEdge } from '../src/types'

function createAdapter() {
  return new BinaryStoreAdapter({
    storeDir: 'test',
    compactThreshold: 1000,
    fileIO: new MemoryFileIO(),
  })
}

describe('BinaryStoreAdapter', () => {
  let adapter: BinaryStoreAdapter

  beforeEach(() => {
    adapter = createAdapter()
  })

  describe('nodes', () => {
    it('putNode and getNode', async () => {
      const node: SerializedNode = { id: 'n1', type: 'doc', data: { x: 1 }, vector: null, insertedAt: 10, updatedAt: 10 }
      await adapter.putNode(node)
      const got = await adapter.getNode('n1')
      expect(got).toBeDefined()
      expect(got!.data).toEqual({ x: 1 })
    })

    it('bulkPutNodes and getNodes', async () => {
      await adapter.bulkPutNodes([
        { id: 'a', type: 't', data: {}, vector: null, insertedAt: 1, updatedAt: 1 },
        { id: 'b', type: 't', data: {}, vector: null, insertedAt: 2, updatedAt: 2 },
      ])
      const nodes = await adapter.getNodes(['a', 'b'])
      expect(nodes).toHaveLength(2)
    })

    it('deleteNode', async () => {
      await adapter.putNode({ id: 'x', type: 't', data: {}, vector: null, insertedAt: 1, updatedAt: 1 })
      await adapter.deleteNode('x')
      expect(await adapter.getNode('x')).toBeUndefined()
    })

    it('bulkDeleteNodes', async () => {
      await adapter.bulkPutNodes([
        { id: 'a', type: 't', data: {}, vector: null, insertedAt: 1, updatedAt: 1 },
        { id: 'b', type: 't', data: {}, vector: null, insertedAt: 2, updatedAt: 2 },
      ])
      await adapter.bulkDeleteNodes(['a', 'b'])
      expect(await adapter.allNodeIds()).toHaveLength(0)
    })

    it('allNodeIds returns all keys', async () => {
      await adapter.bulkPutNodes([
        { id: 'a', type: 't', data: {}, vector: null, insertedAt: 1, updatedAt: 1 },
        { id: 'b', type: 't', data: {}, vector: null, insertedAt: 2, updatedAt: 2 },
      ])
      const ids = await adapter.allNodeIds()
      expect(ids.sort()).toEqual(['a', 'b'])
    })

    it('queries nodes by type, attributes, ranges, and order', async () => {
      await adapter.bulkPutNodes([
        { id: 'a', type: 'book', data: { genre: 'sci-fi', price: 20 }, vector: null, insertedAt: 1, updatedAt: 1 },
        { id: 'b', type: 'book', data: { genre: 'sci-fi', price: 10 }, vector: null, insertedAt: 2, updatedAt: 2 },
        { id: 'c', type: 'book', data: { genre: 'fantasy', price: 15 }, vector: null, insertedAt: 3, updatedAt: 3 },
        { id: 'd', type: 'user', data: { genre: 'sci-fi', price: 12 }, vector: null, insertedAt: 4, updatedAt: 4 },
      ])
      expect(adapter.queryNodes).toBeDefined()

      const nodes = await adapter.queryNodes!({
        nodeTypes: ['book'],
        attributes: { genre: 'sci-fi' },
        attributeRanges: { price: { above: 5, below: 25 } },
        orderBy: { field: 'price', direction: 'asc' },
      })

      expect(nodes.map(node => node.id)).toEqual(['b', 'a'])
      expect(await adapter.countNodes!({ nodeTypes: ['book'] })).toBe(3)
      expect((await adapter.queryNodes!({ nodeTypes: ['book'], offset: 1, limit: 1 })).map(node => node.id)).toEqual(['b'])
      expect(await adapter.countNodes!({ nodeTypes: ['book'], offset: 1, limit: 1 })).toBe(1)
    })
  })

  describe('atomic changes', () => {
    it('applies mixed node, edge, and vector changes together', async () => {
      expect(adapter.applyChanges).toBeDefined()
      await adapter.applyChanges!({
        putNodes: [
          { id: 'a', type: 't', data: {}, vector: [1, 0], insertedAt: 1, updatedAt: 1 },
          { id: 'b', type: 't', data: {}, vector: null, insertedAt: 2, updatedAt: 2 },
        ],
        deleteNodeIds: [],
        putEdges: [{ id: 'a::REL::b', source: 'a', target: 'b', type: 'REL', data: null, createdAt: 1 }],
        deleteEdgeIds: [],
        putVectors: [{ id: 'a', vector: [1, 0] }],
        deleteVectorIds: [],
      })

      expect(await adapter.allNodeIds()).toEqual(expect.arrayContaining(['a', 'b']))
      expect(await adapter.getAllEdges()).toHaveLength(1)
      expect(await adapter.getAllVectors()).toEqual([{ id: 'a', vector: [1, 0] }])

      await adapter.applyChanges!({
        putNodes: [], deleteNodeIds: ['a'], putEdges: [],
        deleteEdgeIds: ['a::REL::b'], putVectors: [], deleteVectorIds: ['a'],
      })
      expect(await adapter.getNode('a')).toBeUndefined()
      expect(await adapter.getAllEdges()).toHaveLength(0)
      expect(await adapter.getAllVectors()).toHaveLength(0)
    })
  })

  describe('edges', () => {
    it('putEdge and getAllEdges', async () => {
      const edge: SerializedEdge = { id: 'a::REL::b', source: 'a', target: 'b', type: 'REL', data: null, createdAt: 100 }
      await adapter.putEdge(edge)
      const all = await adapter.getAllEdges()
      expect(all).toHaveLength(1)
      expect(all[0].source).toBe('a')
    })

    it('bulkPutEdges', async () => {
      await adapter.bulkPutEdges([
        { id: 'a::R::b', source: 'a', target: 'b', type: 'R', data: null, createdAt: 1 },
        { id: 'b::R::c', source: 'b', target: 'c', type: 'R', data: null, createdAt: 2 },
      ])
      expect(await adapter.getAllEdges()).toHaveLength(2)
    })

    it('deleteEdge', async () => {
      await adapter.putEdge({ id: 'a::R::b', source: 'a', target: 'b', type: 'R', data: null, createdAt: 1 })
      await adapter.deleteEdge('a::R::b')
      expect(await adapter.getAllEdges()).toHaveLength(0)
    })

    it('looks up edges by indexed source and target', async () => {
      await adapter.bulkPutEdges([
        { id: 'a::R::b', source: 'a', target: 'b', type: 'R', data: null, createdAt: 1 },
        { id: 'a::S::c', source: 'a', target: 'c', type: 'S', data: null, createdAt: 2 },
        { id: 'd::R::b', source: 'd', target: 'b', type: 'R', data: null, createdAt: 3 },
      ])
      expect(adapter.getEdgesBySources).toBeDefined()
      expect(adapter.getEdgesByTargets).toBeDefined()

      expect((await adapter.getEdgesBySources!(['a'], 'R')).map(edge => edge.id)).toEqual(['a::R::b'])
      expect((await adapter.getEdgesByTargets!(['b'], 'R')).map(edge => edge.id).sort()).toEqual(['a::R::b', 'd::R::b'])
    })
  })

  describe('vectors', () => {
    it('putVector and getAllVectors', async () => {
      await adapter.putVector('v1', [1, 2, 3])
      const all = await adapter.getAllVectors()
      expect(all).toHaveLength(1)
      expect(all[0].vector).toEqual([1, 2, 3])
    })

    it('bulkPutVectors', async () => {
      await adapter.bulkPutVectors([
        { id: 'a', vector: [1, 0] },
        { id: 'b', vector: [0, 1] },
      ])
      expect(await adapter.getAllVectors()).toHaveLength(2)
    })

    it('deleteVector and getVectors', async () => {
      await adapter.putVector('v1', [1, 2, 3])
      await adapter.putVector('v2', [4, 5, 6])
      expect(await adapter.getVectors(['v2'])).toEqual([{ id: 'v2', vector: [4, 5, 6] }])
      await adapter.deleteVector('v1')
      expect(await adapter.getAllVectors()).toHaveLength(1)
    })
  })

  describe('lifecycle', () => {
    it('clearAll removes everything', async () => {
      await adapter.putNode({ id: 'n', type: 't', data: {}, vector: null, insertedAt: 1, updatedAt: 1 })
      await adapter.putEdge({ id: 'e', source: 'a', target: 'b', type: 'R', data: null, createdAt: 1 })
      await adapter.putVector('v', [1, 2, 3])
      await adapter.clearAll()
      expect(await adapter.allNodeIds()).toHaveLength(0)
      expect(await adapter.getAllEdges()).toHaveLength(0)
      expect(await adapter.getAllVectors()).toHaveLength(0)
    })

    it('close does not throw', async () => {
      await adapter.close()
    })
  })

  describe('persistence across instances', () => {
    it('survives close and reopen with snapshot', async () => {
      const io = new MemoryFileIO()
      const a = new BinaryStoreAdapter({ storeDir: 'test', fileIO: io })
      await a.putNode({ id: 'p1', type: 't', data: { v: 1 }, vector: null, insertedAt: 1, updatedAt: 1 })
      await a.putEdge({ id: 'a::REL::b', source: 'a', target: 'b', type: 'REL', data: null, createdAt: 1 })
      await a.putVector('v1', [1, 2, 3])
      await a.close()

      const b = new BinaryStoreAdapter({ storeDir: 'test', fileIO: io })
      const node = await b.getNode('p1')
      expect(node).toBeDefined()
      expect(node!.data).toEqual({ v: 1 })
      const edges = await b.getAllEdges()
      expect(edges).toHaveLength(1)
      const vectors = await b.getAllVectors()
      expect(vectors).toHaveLength(1)
      expect(vectors[0].vector).toEqual([1, 2, 3])
      await b.close()
    })

    it('replays WAL entries on open after snapshot', async () => {
      const io = new MemoryFileIO()
      const a = new BinaryStoreAdapter({ storeDir: 'test', compactThreshold: 2, fileIO: io })
      await a.putNode({ id: 'n1', type: 't', data: {}, vector: null, insertedAt: 1, updatedAt: 1 })
      // This triggers compact (WAL >= 2 entries -> snapshot written, WAL cleared)
      await a.bulkPutNodes([
        { id: 'n2', type: 't', data: {}, vector: null, insertedAt: 2, updatedAt: 2 },
      ])
      // Wait for debounced compact
      await new Promise(r => setTimeout(r, 200))
      // Now write more entries without triggering compact yet
      await a.bulkPutNodes([
        { id: 'n3', type: 't', data: {}, vector: null, insertedAt: 3, updatedAt: 3 },
        { id: 'n4', type: 't', data: {}, vector: null, insertedAt: 4, updatedAt: 4 },
      ])
      await a.close()

      const b = new BinaryStoreAdapter({ storeDir: 'test', compactThreshold: 2, fileIO: io })
      const ids = await b.allNodeIds()
      expect(ids.sort()).toEqual(['n1', 'n2', 'n3', 'n4'])
      await b.close()
    })

    it('handles clean start with no files', async () => {
      const io = new MemoryFileIO()
      const a = new BinaryStoreAdapter({ storeDir: 'empty', fileIO: io })
      expect(await a.allNodeIds()).toHaveLength(0)
      expect(await a.getAllEdges()).toHaveLength(0)
      expect(await a.getAllVectors()).toHaveLength(0)
      await a.close()
    })

    it('bulk vector ops persist across restarts', async () => {
      const io = new MemoryFileIO()
      const a = new BinaryStoreAdapter({ storeDir: 'test', fileIO: io })
      await a.bulkPutVectors([
        { id: 'x', vector: [9, 8, 7] },
        { id: 'y', vector: [6, 5, 4] },
      ])
      await a.close()

      const b = new BinaryStoreAdapter({ storeDir: 'test', fileIO: io })
      const all = await b.getAllVectors()
      expect(all).toHaveLength(2)
      expect(all.find(v => v.id === 'x')!.vector).toEqual([9, 8, 7])
      await b.close()
    })
  })
})

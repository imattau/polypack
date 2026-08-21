import { describe, expect, it } from 'vitest'
import { MemoryAdapter, PolyGraph, QueryAbortedError, QueryLimitError } from '../src/index'

const node = (id: string, order: number) => ({
  id,
  type: 'record',
  data: { order },
  insertedAt: 1,
  updatedAt: 1,
})

describe('persisted query resource limits', () => {
  it('rejects queries that exceed the result limit', async () => {
    const graph = new PolyGraph(new MemoryAdapter())
    graph.addNode(node('a', 1))
    graph.addNode(node('b', 2))
    graph.addNode(node('c', 3))
    await graph.flush()

    await expect(graph.queryPersisted({ maxResults: 2 }).toArray())
      .rejects.toBeInstanceOf(QueryLimitError)
  })

  it('rejects traversal depth and visited-node limits', async () => {
    const graph = new PolyGraph(new MemoryAdapter())
    graph.addNode(node('a', 1))
    graph.addNode(node('b', 2))
    graph.addNode(node('c', 3))
    graph.addEdge('a', 'NEXT', 'b')
    graph.addEdge('b', 'NEXT', 'c')
    await graph.flush()

    await expect(graph.queryPersisted({ maxTraversalDepth: 1 })
      .where('order', 1).traverse('NEXT', 2).toArray())
      .rejects.toBeInstanceOf(QueryLimitError)
    await expect(graph.queryPersisted({ maxNodesVisited: 2 })
      .where('order', 1).traverse('NEXT', 2).toArray())
      .rejects.toBeInstanceOf(QueryLimitError)
  })

  it('rejects an already-aborted query and preserves unlimited behavior by default', async () => {
    const graph = new PolyGraph(new MemoryAdapter())
    graph.addNode(node('a', 1))
    graph.addNode(node('b', 2))
    await graph.flush()

    const controller = new AbortController()
    controller.abort()
    await expect(graph.queryPersisted({ signal: controller.signal }).toArray())
      .rejects.toBeInstanceOf(QueryAbortedError)
    await expect(graph.queryPersisted().whereNodeType('record').ids()).resolves.toEqual(['a', 'b'])
  })

  it('propagates cancellation into direct persistence adapter scans', async () => {
    const adapter = new MemoryAdapter()
    await adapter.putNode(node('a', 1))
    const controller = new AbortController()
    controller.abort()
    await expect(adapter.queryNodes!({ signal: controller.signal })).rejects.toBeInstanceOf(QueryAbortedError)
  })
})

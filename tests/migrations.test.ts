import { describe, expect, it } from 'vitest'
import { MemoryAdapter, MigrationError, PolyGraph } from '../src/index'

describe('application migrations', () => {
  it('runs contiguous node migrations in batches with progress reporting', async () => {
    const graph = new PolyGraph(new MemoryAdapter())
    graph.addNode({ id: 'a', type: 'record', data: { name: 'Alice' }, insertedAt: 1, updatedAt: 1 })
    graph.addNode({ id: 'b', type: 'record', data: { name: 'Bob' }, insertedAt: 1, updatedAt: 1 })
    await graph.flush()
    graph.migrations.register({ from: 1, to: 2, migrateNode: node => ({ ...node, data: { ...node.data, displayName: node.data.name } }) })
    graph.migrations.register({ from: 2, to: 3, migrateNode: node => ({ ...node, data: { ...node.data, migrated: true } }) })
    const progress: number[] = []

    const report = await graph.migrations.run(graph, 1, 3, {
      batchSize: 1,
      onProgress: update => progress.push(update.processed),
    })

    expect(report).toMatchObject({ from: 1, to: 3, processed: 2, total: 2, migrated: 2, dryRun: false })
    expect(progress).toEqual([1, 2])
    expect((await graph.queryPersisted().ids()).sort()).toEqual(['a', 'b'])
    expect((await graph.queryPersisted().first())?.data.migrated).toBe(true)
  })

  it('supports dry runs without mutating records and rejects missing paths', async () => {
    const graph = new PolyGraph(new MemoryAdapter())
    graph.addNode({ id: 'a', type: 'record', data: { old: true }, insertedAt: 1, updatedAt: 1 })
    await graph.flush()
    graph.migrations.register({ from: 1, to: 2, migrateNode: node => ({ ...node, data: { changed: true } }) })

    const report = await graph.migrations.run(graph, 1, 2, { dryRun: true })
    expect(report.dryRun).toBe(true)
    expect((await graph.queryPersisted().first())?.data).toEqual({ old: true })
    await expect(graph.migrations.run(graph, 2, 4)).rejects.toThrow(MigrationError)
  })

  it('migrates edge metadata in the same migration batches', async () => {
    const graph = new PolyGraph(new MemoryAdapter())
    graph.addNode({ id: 'a', type: 'record', data: {}, insertedAt: 1, updatedAt: 1 })
    graph.addNode({ id: 'b', type: 'record', data: {}, insertedAt: 1, updatedAt: 1 })
    graph.addEdge({ id: 'claim', source: 'a', target: 'b', type: 'RELATED', data: { legacy: true }, createdAt: 1 })
    await graph.flush()
    graph.migrations.register({
      from: 1,
      to: 2,
      migrateNode: node => node,
      migrateEdge: edge => ({ ...edge, data: { ...edge.data, migrated: true } }),
    })

    const report = await graph.migrations.run(graph, 1, 2)
    expect(report.total).toBe(3)
    expect(graph.getEdges('a')[0].data).toMatchObject({ legacy: true, migrated: true })
  })

  it('returns a resume cursor and skips completed records', async () => {
    const graph = new PolyGraph(new MemoryAdapter())
    graph.addNode({ id: 'a', type: 'record', data: { old: true }, insertedAt: 1, updatedAt: 1 })
    graph.addNode({ id: 'b', type: 'record', data: { old: true }, insertedAt: 1, updatedAt: 1 })
    await graph.flush()
    graph.migrations.register({ from: 1, to: 2, migrateNode: node => ({ ...node, data: { migrated: true } }) })
    const controller = new AbortController()
    let cursor: { kind: 'node' | 'edge'; id: string } | undefined
    await expect(graph.migrations.run(graph, 1, 2, {
      batchSize: 1,
      signal: controller.signal,
      onProgress: progress => { cursor = progress.lastProcessed; controller.abort() },
    })).rejects.toThrow(MigrationError)

    expect(cursor).toEqual({ kind: 'node', id: 'a' })
    const resumed = await graph.migrations.run(graph, 1, 2, { resumeAfter: { nodeId: cursor!.id } })
    expect(resumed.processed).toBe(1)
    expect((await graph.queryPersisted().toArray()).every(node => node.data.migrated === true)).toBe(true)
  })
})

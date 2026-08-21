import { describe, expect, it } from 'vitest'
import { PolyGraph, UniqueConstraintError } from '../src/index'
import { BinaryStoreAdapter } from '../src/persistence/binary-store'
import { MemoryAdapter } from '../src/persistence/memory'
import { MemoryFileIO } from '../src/persistence/file-io'

const node = (id: string, data: Record<string, unknown>) => ({
  id, type: 'person', data, insertedAt: 1, updatedAt: 1,
})

describe('secondary indexes', () => {
  it('enforces unique and compound keys before mutation', () => {
    const graph = new PolyGraph()
    graph.defineIndex({ name: 'external-id', nodeType: 'person', fields: ['provider', 'externalId'], unique: true })
    graph.addNode(node('a', { provider: 'github', externalId: '1' }))

    expect(() => graph.addNode(node('b', { provider: 'github', externalId: '1' }))).toThrow(UniqueConstraintError)
    expect(graph.getNode('b')).toBeUndefined()
  })

  it('updates index membership when a node is patched', () => {
    const graph = new PolyGraph()
    graph.defineIndex({ name: 'email', fields: ['email'], unique: true })
    graph.addNode(node('a', { email: 'a@example.test' }))
    graph.addNode(node('b', { email: 'b@example.test' }))

    expect(() => graph.patchNode('b', { set: { 'data.email': 'a@example.test' } })).toThrow(UniqueConstraintError)
    expect(graph.getNode('b')?.data.email).toBe('b@example.test')
  })

  it('rebuilds secondary index buckets on demand', () => {
    const graph = new PolyGraph()
    graph.defineIndex({ name: 'email', fields: ['email'], unique: true })
    graph.addNode(node('a', { email: 'a@example.test' }))
    graph.rebuildIndexes()

    expect(() => graph.addNode(node('b', { email: 'a@example.test' }))).toThrow(UniqueConstraintError)
  })

  it('rebuilds indexes from persisted records outside the hot cache', async () => {
    const graph = new PolyGraph(new MemoryAdapter())
    graph.defineIndex({ name: 'email', fields: ['email'], unique: true })
    graph.addNode(node('a', { email: 'a@example.test' }))
    await graph.flush()
    graph.clear()
    await graph.rebuildIndexesFromPersistence()

    expect(() => graph.addNode(node('b', { email: 'a@example.test' }))).toThrow(UniqueConstraintError)
  })

  it('preserves the prior index state when a rebuild fails', async () => {
    const adapter = new MemoryAdapter()
    const graph = new PolyGraph(adapter)
    graph.defineIndex({ name: 'email', fields: ['email'], unique: true })
    graph.addNode(node('a', { email: 'a@example.test' }))
    await graph.flush()
    await adapter.putNode({ id: 'b', type: 'person', data: { email: 'a@example.test' }, vector: null, insertedAt: 1, updatedAt: 1 })

    await expect(graph.rebuildIndexesFromPersistence()).rejects.toBeInstanceOf(UniqueConstraintError)
    expect(() => graph.addNode(node('c', { email: 'a@example.test' }))).toThrow(UniqueConstraintError)
  })

  it('supports sparse indexes and explain output', () => {
    const graph = new PolyGraph()
    graph.defineIndex({ name: 'birth-year', nodeType: 'person', fields: ['birthYear'], sparse: true })
    graph.addNode(node('a', { birthYear: 1980 }))
    graph.addNode(node('b', {}))

    const plan = graph.query().whereNodeType('person').where('birthYear', 1980).explain()
    expect(plan.index).toBe('birth-year')
    expect(plan.stages[0]).toBe('property-index(birth-year)')
    expect(plan.loadedRecords).toBe(2)
  })

  it('reports in-memory query metrics through graph stats', async () => {
    const graph = new PolyGraph()
    graph.defineIndex({ name: 'birth-year', fields: ['birthYear'] })
    graph.addNode(node('a', { birthYear: 1980 }))
    graph.query().where('birthYear', 1980).toArray()

    const stats = await graph.stats()
    expect(stats.queryCount).toBe(1)
    expect(stats.queryIndexUsage?.['birth-year']).toBe(1)
    expect(stats.queryScannedRecords).toBe(1)
  })

  it('records every persisted index used by an intersection', async () => {
    const graph = new PolyGraph()
    graph.defineIndex({ name: 'surname', fields: ['surname'] })
    graph.defineIndex({ name: 'birth-year', fields: ['birthYear'] })
    graph.addNode({ id: 'a', type: 'person', data: { surname: 'Smith', birthYear: 1980 }, insertedAt: 1, updatedAt: 1 })
    await graph.flush()
    await graph.queryPersisted().where('surname', 'Smith').where('birthYear', 1980).toArray()
    expect((await graph.stats()).queryIndexUsage).toEqual({ surname: 1, 'birth-year': 1 })
  })

  it('uses a numeric secondary index for range candidates', () => {
    const graph = new PolyGraph()
    graph.defineIndex({ name: 'birth-year', nodeType: 'person', fields: ['birthYear'], sparse: true })
    graph.addNode(node('a', { birthYear: 1980 }))
    graph.addNode(node('b', { birthYear: 2020 }))
    graph.addNode(node('c', { birthYear: 2050 }))

    expect(graph.query().whereNodeType('person').whereAttributeRange('birthYear', { above: 2000 }).toArray().map(n => n.id)).toEqual(['b', 'c'])
  })

  it('explains persisted index selection and traversal stages', async () => {
    const graph = new PolyGraph()
    graph.defineIndex({ name: 'person-surname', nodeType: 'person', fields: ['surname'] })
    graph.addNode({ id: 'a', type: 'person', data: { surname: 'Smith' }, insertedAt: 1, updatedAt: 1 })
    graph.addNode({ id: 'b', type: 'person', data: { surname: 'Jones' }, insertedAt: 1, updatedAt: 1 })
    await graph.flush()

    const plan = await graph.queryPersisted()
      .whereNodeType('person')
      .where('surname', 'Smith')
      .traverse('RELATED', 2)
      .explain()
    expect(plan.index).toBe('person-surname')
    expect(plan.stages).toContain('property-index(person-surname)')
    expect(plan.stages).toContain('traversal(depth=2)')
    expect(plan.loadedRecords).toBe(2)
  })

  it('intersects persisted secondary indexes for compound predicates and ranges', async () => {
    const graph = new PolyGraph()
    graph.defineIndex({ name: 'surname', nodeType: 'person', fields: ['surname'] })
    graph.defineIndex({ name: 'birth-year', nodeType: 'person', fields: ['birthYear'], sparse: true })
    graph.addNode({ id: 'match', type: 'person', data: { surname: 'Smith', birthYear: 1980 }, insertedAt: 1, updatedAt: 1 })
    graph.addNode({ id: 'wrong-name', type: 'person', data: { surname: 'Jones', birthYear: 1980 }, insertedAt: 1, updatedAt: 1 })
    graph.addNode({ id: 'wrong-year', type: 'person', data: { surname: 'Smith', birthYear: 2020 }, insertedAt: 1, updatedAt: 1 })
    await graph.flush()

    const query = graph.queryPersisted()
      .whereNodeType('person')
      .where('surname', 'Smith')
      .whereAttributeRange('birthYear', { below: 2000 })
    expect(await query.ids()).toEqual(['match'])
    const plan = await query.explain()
    expect(plan.indexes).toEqual(['surname', 'birth-year'])
    expect(plan.stages).toContain('index-intersection(2)')
  })

  it('persists index definitions in the binary store', async () => {
    const io = new MemoryFileIO()
    const graph = new PolyGraph(new BinaryStoreAdapter({ storeDir: 'index-metadata', fileIO: io }))
    graph.defineIndex({ name: 'email', nodeType: 'person', fields: ['email'], unique: true })
    await graph.flush()

    const reopened = new PolyGraph(new BinaryStoreAdapter({ storeDir: 'index-metadata', fileIO: io }))
    await reopened.warm()
    expect(reopened.indexes).toEqual([{ name: 'email', nodeType: 'person', fields: ['email'], unique: true }])
  })
})

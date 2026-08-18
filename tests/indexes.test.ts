import { describe, expect, it } from 'vitest'
import { PolyGraph, UniqueConstraintError } from '../src/index'
import { BinaryStoreAdapter } from '../src/persistence/binary-store'
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

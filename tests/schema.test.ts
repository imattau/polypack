import { describe, expect, it } from 'vitest'
import { PolyGraph, SchemaValidationError, UniqueConstraintError } from '../src/index'

const person = (id: string, name: string) => ({
  id,
  type: 'person',
  data: { name },
  insertedAt: 1,
  updatedAt: 1,
})

describe('schema and constraint hooks', () => {
  it('validates node types before mutation and can define convenience indexes', () => {
    const graph = new PolyGraph()
    graph.registerNodeType('person', {
      indexes: ['name'],
      validate: node => typeof node.data.name === 'string' && node.data.name.length > 0,
    })

    graph.addNode(person('alice', 'Alice'))
    expect(graph.query().where('name', 'Alice').ids()).toEqual(['alice'])
    expect(() => graph.addNode({ ...person('invalid', ''), data: { name: '' } }))
      .toThrow(SchemaValidationError)
    expect(graph.getNode('invalid')).toBeUndefined()
    expect(() => graph.updateNode('alice', { name: '' })).toThrow(SchemaValidationError)
    expect(graph.getNode('alice')?.data.name).toBe('Alice')
  })

  it('enforces endpoint types, referential integrity, custom edge validation, and cardinality', () => {
    const graph = new PolyGraph()
    graph.registerNodeType('person')
    graph.registerNodeType('group')
    graph.registerEdgeType('MEMBER_OF', {
      sourceTypes: ['person'],
      targetTypes: ['group'],
      cardinality: 'many-to-one',
      validate: edge => edge.data?.role !== undefined,
    })
    graph.addNode(person('alice', 'Alice'))
    graph.addNode({ id: 'team', type: 'group', data: {}, insertedAt: 1, updatedAt: 1 })
    graph.addNode({ id: 'other-team', type: 'group', data: {}, insertedAt: 1, updatedAt: 1 })
    graph.addNode(person('bob', 'Bob'))

    expect(() => graph.addEdge({ id: 'bad', source: 'alice', target: 'team', type: 'MEMBER_OF', data: {}, createdAt: 1 }))
      .toThrow(SchemaValidationError)
    graph.addEdge({ id: 'alice-team', source: 'alice', target: 'team', type: 'MEMBER_OF', data: { role: 'owner' }, createdAt: 1 })
    graph.addEdge({ id: 'bob-team', source: 'bob', target: 'team', type: 'MEMBER_OF', data: { role: 'member' }, createdAt: 1 })
    expect(graph.getEdges('alice', 'MEMBER_OF')).toHaveLength(1)
    expect(() => graph.addEdge({ id: 'alice-other-team', source: 'alice', target: 'other-team', type: 'MEMBER_OF', data: { role: 'member' }, createdAt: 1 }))
      .toThrow(SchemaValidationError)
    expect(() => graph.addEdge({ id: 'missing', source: 'alice', target: 'unknown', type: 'MEMBER_OF', data: { role: 'member' }, createdAt: 1 }))
      .toThrow(SchemaValidationError)
  })

  it('enforces unique indexes together with schema registration', () => {
    const graph = new PolyGraph()
    graph.registerNodeType('person', { indexes: ['name'] })
    graph.defineIndex({ name: 'unique-person-name', nodeType: 'person', fields: ['name'], unique: true })
    graph.addNode(person('alice', 'Alice'))
    expect(() => graph.addNode(person('duplicate', 'Alice'))).toThrow(UniqueConstraintError)
  })
})

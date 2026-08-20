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

  it('supports declarative required fields and data types on edges', () => {
    const graph = new PolyGraph()
    graph.registerNodeType('person')
    graph.registerEdgeType('KNOWS', { requiredFields: ['since'], dataTypes: { since: 'integer' } })
    graph.addNode(person('a', 'A'))
    graph.addNode(person('b', 'B'))
    expect(() => graph.addEdge({ id: 'missing', source: 'a', target: 'b', type: 'KNOWS', data: {}, createdAt: 1 })).toThrow(SchemaValidationError)
    expect(() => graph.addEdge({ id: 'wrong', source: 'a', target: 'b', type: 'KNOWS', data: { since: 'old' }, createdAt: 1 })).toThrow(SchemaValidationError)
    graph.addEdge({ id: 'valid', source: 'a', target: 'b', type: 'KNOWS', data: { since: 2020 }, createdAt: 1 })
    expect(graph.getEdges('a', 'KNOWS')).toHaveLength(1)
  })

  it('enforces unique indexes together with schema registration', () => {
    const graph = new PolyGraph()
    graph.registerNodeType('person', { indexes: ['name'] })
    graph.defineIndex({ name: 'unique-person-name', nodeType: 'person', fields: ['name'], unique: true })
    graph.addNode(person('alice', 'Alice'))
    expect(() => graph.addNode(person('duplicate', 'Alice'))).toThrow(UniqueConstraintError)
  })

  it('supports declarative required fields and data types', () => {
    const graph = new PolyGraph()
    graph.registerNodeType('person', {
      requiredFields: ['name', 'birthYear'],
      dataTypes: { name: 'string', birthYear: 'integer' },
    })
    expect(() => graph.addNode({ id: 'missing', type: 'person', data: { name: 'A' }, insertedAt: 1, updatedAt: 1 }))
      .toThrow(SchemaValidationError)
    expect(() => graph.addNode({ id: 'wrong', type: 'person', data: { name: 'A', birthYear: 'old' }, insertedAt: 1, updatedAt: 1 }))
      .toThrow(SchemaValidationError)
    graph.addNode({ id: 'valid', type: 'person', data: { name: 'A', birthYear: 2000 }, insertedAt: 1, updatedAt: 1 })
    expect(graph.getNode('valid')).toBeDefined()
  })

  it('returns defensive schema definitions', () => {
    const graph = new PolyGraph()
    graph.registerNodeType('person', { indexes: ['name'], requiredFields: ['name'] })
    graph.registerEdgeType('KNOWS', { sourceTypes: ['person'] })
    graph.nodeTypes.get('person')!.indexes!.push('mutated')
    graph.edgeTypes.get('KNOWS')!.sourceTypes!.push('mutated')
    expect(graph.nodeTypes.get('person')?.indexes).toEqual(['name'])
    expect(graph.edgeTypes.get('KNOWS')?.sourceTypes).toEqual(['person'])
  })

  it('rejects malformed schema definitions before registration mutation', () => {
    const graph = new PolyGraph()
    expect(() => graph.registerNodeType('person', { requiredFields: ['name', 'name'] })).toThrow(/unique and non-empty/)
    expect(() => graph.registerNodeType('person', { dataTypes: { age: 'date' as never } })).toThrow(/Invalid node data type/)
    expect(() => graph.registerEdgeType('RELATED', { sourceTypes: ['person', 'person'] })).toThrow(/unique and non-empty/)
    expect(() => graph.registerEdgeType('RELATED', { cardinality: 'invalid' as never })).toThrow(/Invalid edge cardinality/)
    expect(graph.nodeTypes.has('person')).toBe(false)
    expect(graph.edgeTypes.has('RELATED')).toBe(false)
  })
})

import { describe, expect, it } from 'vitest'
import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { PolyGraph } from '../src/graph'

const fixture = JSON.parse(readFileSync(join(process.cwd(), 'fixtures/database-core/schema-and-indexes.json'), 'utf8'))

describe('schema and index conformance fixture', () => {
  it('validates before mutation and enforces uniqueness', () => {
    const graph = new PolyGraph()
    graph.registerNodeType(fixture.nodeType.name, fixture.nodeType)
    graph.defineIndex(fixture.index)
    graph.addNode(fixture.validNode)
    expect(() => graph.addNode(fixture.invalidNode)).toThrow()
    expect(() => graph.addNode(fixture.duplicateNode)).toThrow()
    expect(graph.size).toBe(fixture.expect.nodeCount)
    expect(graph.getNode(fixture.expect.presentId)?.data.name).toBe('Mary')
  })
})

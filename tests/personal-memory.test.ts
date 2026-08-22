import { describe, expect, it } from 'vitest'
import { createPersonalMemory, recall, remember } from '../examples/personal-memory'

describe('personal memory reference workflow', () => {
  it('retrieves context, preserves provenance, and learns corrections', async () => {
    const memory = createPersonalMemory()
    await remember(memory, 'old', 'The old answer says Polypack is only a graph database.', {
      type: 'fact', memoryClass: 'semantic', confidence: 0.4, source: 'user',
    })
    await remember(memory, 'new', 'The corrected answer says Polypack combines graph, vector search, and adaptive memory.', {
      type: 'fact', memoryClass: 'semantic', confidence: 0.95, source: 'project:README',
    })
    memory.graph.supersede('new', 'old', 1, 'corrected')

    const results = await recall(memory, 'graph vector adaptive memory', 'project')

    expect(results.map(node => node.id)).toContain('new')
    expect(memory.graph.getNode('new')?.source).toBe('project:README')
    expect(memory.graph.getNode('new')?.supersedes).toBe('old')
    expect(memory.graph.getNode('old')?.activation?.inhibition).toBeGreaterThan(0)
    expect(memory.graph.getEdgeTargets('new', 'SUPERSEDED_BY')).toEqual(['old'])

    memory.engine.dispose()
    await memory.graph.dispose()
  })

  it('consolidates related memories without deleting evidence', async () => {
    const memory = createPersonalMemory()
    await remember(memory, 'a', 'Local storage keeps data available offline.', { source: 'note:a' })
    await remember(memory, 'b', 'Durable sync carries changes between devices.', { source: 'note:b' })

    const summary = memory.graph.consolidate({
      id: 'summary',
      type: 'summary',
      data: { text: 'Polypack is local-first and synchronizable.' },
      memoryClass: 'semantic',
      confidence: 0.9,
      source: 'inference:test',
      insertedAt: Date.now(),
      updatedAt: Date.now(),
    }, ['a', 'b'])

    expect(summary?.derivedFrom).toEqual(['a', 'b'])
    expect(memory.graph.getNode('a')).toBeDefined()
    expect(memory.graph.getNode('b')).toBeDefined()
    expect(memory.graph.getEdgeTargets('summary', 'CONSOLIDATED_FROM')).toEqual(['a', 'b'])

    memory.engine.dispose()
    await memory.graph.dispose()
  })
})


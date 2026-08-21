import { describe, it, expect, vi, afterEach } from 'vitest'
import { PolyGraph } from '../src/graph'
import { MemoryAdapter } from '../src/persistence/memory'
import { BinaryStoreAdapter } from '../src/persistence/binary-store'
import { MemoryFileIO } from '../src/persistence/file-io'
import { FeatureHashEmbedding } from '../src/embedding'
import { ActivationEngine, mergeActivation } from '../src/activation'
import { decayFactor } from '../src/utils'
import type { NodeActivation } from '../src/types'
import { SyncServer } from '../src/sync/server'
import { SyncClient } from '../src/sync/client'
import { MemoryTransport } from '../src/sync/transport'

const HOUR = 3_600_000
const DAY = 24 * HOUR

function node(id: string, insertedAt = 1, activation?: NodeActivation) {
  return {
    id,
    type: 't',
    data: {},
    insertedAt,
    updatedAt: insertedAt,
    ...(activation ? { activation } : {}),
  }
}

const settle = () => new Promise<void>((resolve) => setTimeout(resolve, 20))

function connect(
  graph: PolyGraph,
  clientId: string,
  server: SyncServer,
  autoFlush = true,
): { client: SyncClient; cleanup: () => void } {
  const [clientT, serverT] = MemoryTransport.pair()
  const client = new SyncClient({ graph, transport: clientT, clientId, autoFlush })
  const onServerMsg = server.addClient({ send: (msg) => client.handleMessage(msg), clientId })
  serverT.onMessage = (msg) => onServerMsg(msg)
  return { client, cleanup: () => client.disconnect() }
}

afterEach(() => {
  vi.useRealTimers()
})

describe('PolyGraph activation', () => {
  it('reinforces a node and emits an activation_updated event', () => {
    const graph = new PolyGraph()
    graph.addNode(node('n1'))

    const events: string[] = []
    graph.changes.subscribe(e => events.push(e.type))

    const updated = graph.reinforceNode('n1', 0.5, 'user_read')!
    expect(updated.activation).toMatchObject({ score: 0.5, importance: 0.025, reinforcementCount: 1 })
    expect(updated.activation!.lastMeaningfulActivation).toBeGreaterThan(0)
    expect(graph.getActivation('n1')).toBeCloseTo(0.5, 5)
    expect(events).toEqual(['activation_updated'])
  })

  it('clamps score and importance to [0, 1]', () => {
    const graph = new PolyGraph()
    graph.addNode(node('n1'))
    graph.reinforceNode('n1', 5)
    expect(graph.getActivationState('n1')!.score).toBeCloseTo(1, 5)
  })

  it('accumulates deltas across reinforcements', () => {
    const graph = new PolyGraph()
    graph.addNode(node('n1'))
    graph.reinforceNode('n1', 0.2)
    graph.reinforceNode('n1', 0.3)
    const state = graph.getActivationState('n1')!
    expect(state.score).toBeCloseTo(0.5, 5)
    expect(state.reinforcementCount).toBe(2)
  })

  it('returns undefined for a node that is not loaded', () => {
    const graph = new PolyGraph()
    expect(graph.reinforceNode('missing', 0.5)).toBeUndefined()
    expect(graph.getActivation('missing')).toBe(0)
  })

  it('rejects malformed activation records on insert', () => {
    const graph = new PolyGraph()
    expect(() => graph.addNode(node('bad', 1, { score: 2, importance: 0, reinforcementCount: 0, lastMeaningfulActivation: 0 })))
      .toThrow(RangeError)
    expect(() => graph.addNode(node('bad', 1, { score: 0, importance: 0, reinforcementCount: -1, lastMeaningfulActivation: 0 })))
      .toThrow(RangeError)
  })

  it('decays lazily as a pure function of elapsed time', () => {
    const graph = new PolyGraph()
    const now = Date.now()
    // Anchored one half-life in the past at score 1 → current score 0.5.
    graph.addNode(node('n1', 1, {
      score: 1, importance: 1, reinforcementCount: 3, lastMeaningfulActivation: now - DAY,
    }))
    expect(graph.getActivation('n1')).toBeCloseTo(0.5, 5)
    expect(graph.getActivation('n1', DAY / 2)).toBeCloseTo(0.25, 5)
    // Two half-lives → 0.25.
    graph.addNode(node('n2', 1, {
      score: 1, importance: 1, reinforcementCount: 3, lastMeaningfulActivation: now - 2 * DAY,
    }))
    expect(graph.getActivation('n2')).toBeCloseTo(0.25, 5)
  })

  it('reinforceNode decay-corrects the prior score before adding', () => {
    const graph = new PolyGraph()
    const now = Date.now()
    graph.addNode(node('n1', 1, {
      score: 1, importance: 0.5, reinforcementCount: 1, lastMeaningfulActivation: now - DAY,
    }))
    // Decayed to 0.5, then +0.5 → 1.0, re-anchored to now.
    const clock = vi.spyOn(Date, 'now').mockReturnValue(now)
    try {
      graph.reinforceNode('n1', 0.5)
    } finally {
      clock.mockRestore()
    }
    const state = graph.getActivationState('n1')!
    expect(state.score).toBeCloseTo(1, 5)
    expect(state.lastMeaningfulActivation).toBe(now)
    expect(state.reinforcementCount).toBe(2)
  })

  it('persists activation through flush and warm', async () => {
    const io = new MemoryFileIO()
    const graph = new PolyGraph(new BinaryStoreAdapter({ storeDir: 'test', compactThreshold: 1000, fileIO: io }))
    graph.addNode(node('n1'))
    graph.reinforceNode('n1', 0.6)
    await graph.flush()
    await graph.dispose()

    const restored = new PolyGraph(new BinaryStoreAdapter({ storeDir: 'test', compactThreshold: 1000, fileIO: io }))
    await restored.warm()
    const loaded = restored.getNode('n1')!
    expect(loaded.activation!.score).toBeCloseTo(0.6, 5)
    expect(loaded.activation!.reinforcementCount).toBe(1)
    await restored.dispose()
  })

  it('restores and reinforces an evicted node via reinforceNodeSafe', async () => {
    const graph = new PolyGraph(new MemoryAdapter(), 1)
    graph.addNode(node('n1'))
    graph.reinforceNode('n1', 0.4)
    await graph.flush()
    graph.addNode(node('n2'))
    await graph.flush()
    expect(graph.hasLoadedNode('n1')).toBe(false)

    const updated = await graph.reinforceNodeSafe('n1', 0.3)!
    expect(updated).toBeDefined()
    expect(updated.activation!.score).toBeCloseTo(0.7, 5)
    expect(graph.hasLoadedNode('n1')).toBe(true)
  })

  it('topActivated ranks loaded nodes by current activation descending', () => {
    const graph = new PolyGraph()
    for (const [id, amount] of [['a', 0.9], ['b', 0.5], ['c', 0.1]] as const) {
      graph.addNode(node(id))
      graph.reinforceNode(id, amount)
    }
    expect(graph.topActivated(2).map(n => n.id)).toEqual(['a', 'b'])
    expect(graph.topActivated(10, 0.6).map(n => n.id)).toEqual(['a'])
  })

  it('supports whereActivated and orderByActivation in GraphQuery', () => {
    const graph = new PolyGraph()
    for (const [id, amount] of [['a', 0.9], ['b', 0.5], ['c', 0.1]] as const) {
      graph.addNode(node(id))
      graph.reinforceNode(id, amount)
    }
    expect(graph.query().whereActivated(0.4).ids()).toEqual(['a', 'b'])
    expect(graph.query().orderByActivation('desc').ids()).toEqual(['a', 'b', 'c'])
    expect(graph.query().orderByActivation('asc').ids()).toEqual(['c', 'b', 'a'])
    expect(graph.query().whereActivated(0.4).orderByActivation('desc').ids()).toEqual(['a', 'b'])
  })

  it('supports whereActivated and orderByActivation in PersistedGraphQuery', async () => {
    const graph = new PolyGraph()
    for (const [id, amount] of [['a', 0.9], ['b', 0.5], ['c', 0.1]] as const) {
      graph.addNode(node(id))
      graph.reinforceNode(id, amount)
    }
    await graph.flush()
    const high = await graph.queryPersisted().whereActivated(0.4).ids()
    expect(high).toEqual(['a', 'b'])
    const ordered = await graph.queryPersisted().orderByActivation('desc').ids()
    expect(ordered).toEqual(['a', 'b', 'c'])
  })
})

describe('decay helpers', () => {
  it('decayFactor halves per half-life', () => {
    expect(decayFactor(0, DAY)).toBe(1)
    expect(decayFactor(DAY, DAY)).toBeCloseTo(0.5, 10)
    expect(decayFactor(2 * DAY, DAY)).toBeCloseTo(0.25, 10)
    expect(decayFactor(1000, Infinity)).toBe(1)
  })
})

describe('mergeActivation', () => {
  it('max-merges decay-corrected totals and re-anchors to now', () => {
    const now = Date.now()
    const a: NodeActivation = { score: 0.6, importance: 0.3, reinforcementCount: 2, lastMeaningfulActivation: now - DAY }
    const b: NodeActivation = { score: 0.9, importance: 0.1, reinforcementCount: 1, lastMeaningfulActivation: now }
    const merged = mergeActivation(a, b, now)
    expect(merged.score).toBeCloseTo(0.9, 5)
    // `a` decayed for a full day on the 30-day importance curve.
    expect(merged.importance).toBeCloseTo(0.3 * decayFactor(DAY, 30 * DAY), 5)
    expect(merged.reinforcementCount).toBe(2)
    expect(merged.lastMeaningfulActivation).toBe(now)
  })

  it('max-merges inhibition and per-context scores', () => {
    const now = Date.now()
    const a: NodeActivation = {
      score: 0.2, importance: 0, reinforcementCount: 1, lastMeaningfulActivation: now,
      inhibition: 0.3, lastInhibitedAt: now,
      context: { 'project-x': { score: 0.7, lastMeaningfulActivation: now } },
    }
    const b: NodeActivation = {
      score: 0.5, importance: 0, reinforcementCount: 2, lastMeaningfulActivation: now,
      inhibition: 0.6, lastInhibitedAt: now,
      context: { 'project-x': { score: 0.4, lastMeaningfulActivation: now }, coding: { score: 0.2, lastMeaningfulActivation: now } },
    }
    const merged = mergeActivation(a, b, now)
    expect(merged.inhibition).toBeCloseTo(0.6, 5)
    expect(merged.context?.['project-x'].score).toBeCloseTo(0.7, 5)
    expect(merged.context?.coding.score).toBeCloseTo(0.2, 5)
  })
})

describe('ActivationEngine', () => {
  it('spreads activation outward with per-hop attenuation', () => {
    const graph = new PolyGraph()
    graph.addNode(node('ai'))
    graph.addNode(node('vector-db'))
    graph.addNode(node('polypack'))
    graph.addNode(node('nostr'))
    graph.addEdge('ai', 'REL', 'vector-db')
    graph.addEdge('vector-db', 'REL', 'polypack')
    graph.addEdge('polypack', 'REL', 'nostr')

    const engine = new ActivationEngine(graph)
    const spread = engine.spread(['ai'], { depth: 2, decay: 0.5 })
    expect(spread.get('vector-db')).toBeCloseTo(0.5, 5)
    expect(spread.get('polypack')).toBeCloseTo(0.25, 5)
    expect(spread.has('nostr')).toBe(false)
    engine.dispose()
  })

  it('restricts spreading to chosen edge types', () => {
    const graph = new PolyGraph()
    graph.addNode(node('a'))
    graph.addNode(node('b'))
    graph.addNode(node('c'))
    graph.addEdge('a', 'MENTIONS', 'b')
    graph.addEdge('a', 'IGNORES', 'c')

    const engine = new ActivationEngine(graph)
    const spread = engine.spread(['a'], { depth: 1, edgeTypes: ['MENTIONS'] })
    expect(spread.get('b')).toBeCloseTo(0.5, 5)
    expect(spread.has('c')).toBe(false)
    engine.dispose()
  })

  it('keeps small transient attention local and promotes past the threshold', () => {
    const graph = new PolyGraph()
    graph.addNode(node('n1'))
    graph.reinforceNode('n1', 0.5)

    const engine = new ActivationEngine(graph)
    expect(engine.effective('n1')).toBeCloseTo(0.5, 5)

    // Sub-threshold attention stays local and never touches durable state.
    engine.bumpAttention('n1', 0.02)
    expect(engine.attentionOf('n1')).toBeCloseTo(0.02, 5)
    expect(engine.effective('n1')).toBeCloseTo(0.52, 5)
    expect(graph.getActivation('n1')).toBeCloseTo(0.5, 5)

    // Accumulating past minReinforceDelta (0.05) promotes into durable activation.
    engine.bumpAttention('n1', 0.05)
    expect(engine.attentionOf('n1')).toBe(0)
    expect(graph.getActivation('n1')).toBeCloseTo(0.57, 5)
    engine.dispose()
  })

  it('workingMemory returns the most active nodes', () => {
    const graph = new PolyGraph()
    for (const [id, amount] of [['a', 0.9], ['b', 0.5], ['c', 0.1]] as const) {
      graph.addNode(node(id))
      graph.reinforceNode(id, amount)
    }
    const engine = new ActivationEngine(graph)
    expect(engine.workingMemory(2).map(n => n.id)).toEqual(['a', 'b'])
    engine.dispose()
  })

  it('reinforces a context independently of the global score', () => {
    const graph = new PolyGraph()
    graph.addNode(node('n1'))
    graph.reinforceNode('n1', 0.2)
    graph.reinforceNode('n1', 0.3, undefined, 'project-x')

    const engine = new ActivationEngine(graph)
    // Global score reflects both reinforcements; the context reflects only its own.
    expect(engine.effective('n1')).toBeCloseTo(0.5, 5)
    expect(engine.effective('n1', 'project-x')).toBeCloseTo(0.3, 5)
    // A context the node has no history in reads cold, even though it's globally active.
    expect(engine.effective('n1', 'coding')).toBe(0)
    expect(graph.getContextActivation('n1', 'project-x')).toBeCloseTo(0.3, 5)
    engine.dispose()
  })

  it('suppress subtracts inhibition from effective() but not from the raw score', () => {
    const graph = new PolyGraph()
    graph.addNode(node('n1'))
    graph.reinforceNode('n1', 0.8)
    graph.suppressNode('n1', 0.5)

    const engine = new ActivationEngine(graph)
    expect(engine.inhibitionOf('n1')).toBeCloseTo(0.5, 5)
    expect(engine.effective('n1')).toBeCloseTo(0.3, 5)
    expect(graph.getActivation('n1')).toBeCloseTo(0.8, 5)

    // Releasing suppression (negative amount) restores effective().
    graph.suppressNode('n1', -0.5)
    expect(engine.effective('n1')).toBeCloseTo(0.8, 5)
    engine.dispose()
  })

  it('workingMemory with a tokenBudget stops once the budget is spent', () => {
    const graph = new PolyGraph()
    for (const [id, amount] of [['a', 0.9], ['b', 0.5], ['c', 0.1]] as const) {
      graph.addNode(node(id))
      graph.reinforceNode(id, amount)
    }
    const engine = new ActivationEngine(graph)
    const selected = engine.workingMemory({ limit: 10, tokenBudget: 2, costOf: () => 1 })
    expect(selected.map(n => n.id)).toEqual(['a', 'b'])
    engine.dispose()
  })

  it('workingMemory with diversityLambda penalises redundant neighbours', () => {
    const graph = new PolyGraph()
    graph.addNode({ ...node('a'), vector: new Float64Array([1, 0]) })
    graph.addNode({ ...node('b'), vector: new Float64Array([1, 0.01]) }) // near-duplicate of a
    graph.addNode({ ...node('c'), vector: new Float64Array([0, 1]) }) // orthogonal
    graph.reinforceNode('a', 0.9)
    graph.reinforceNode('b', 0.85)
    graph.reinforceNode('c', 0.5)

    const engine = new ActivationEngine(graph)
    const relevanceOnly = engine.workingMemory({ limit: 2 })
    expect(relevanceOnly.map(n => n.id)).toEqual(['a', 'b'])

    const diverse = engine.workingMemory({ limit: 2, diversityLambda: 0.8 })
    expect(diverse.map(n => n.id)).toEqual(['a', 'c'])
    engine.dispose()
  })

  it('pulse scores a region and absorb reinforces above threshold', async () => {
    const graph = new PolyGraph(undefined, undefined, new FeatureHashEmbedding(32))
    await graph.addNodeWithEmbedding(node('ai-article', 1), 'vector database nearest neighbor search')
    await graph.addNodeWithEmbedding(node('ai-followup', 1), 'vector index query embeddings')
    await graph.addNodeWithEmbedding(node('unrelated', 1), 'grocery shopping list recipes cooking')

    const engine = new ActivationEngine(graph)
    const scores = await engine.pulse('vector search database')
    expect([...scores.keys()]).toContain('ai-article')
    expect([...scores.keys()]).toContain('ai-followup')

    await engine.absorb('vector search database')
    const top = engine.workingMemory(1)
    expect(top.map(n => n.id)).toContain('ai-article')
    engine.dispose()
  })

  it('clears transient attention for removed nodes', () => {
    const graph = new PolyGraph()
    graph.addNode(node('n1'))
    const engine = new ActivationEngine(graph)
    engine.bumpAttention('n1', 0.02)
    expect(engine.attentionOf('n1')).toBeCloseTo(0.02, 5)
    graph.removeNode('n1')
    expect(engine.attentionOf('n1')).toBe(0)
    engine.dispose()
  })
})

describe('activation sync', () => {
  it('accumulates concurrent reinforcement deltas across replicas (0.2 + 0.3 = 0.5)', async () => {
    const server = new SyncServer()
    const aGraph = new PolyGraph()
    const bGraph = new PolyGraph()
    const { cleanup: ca } = connect(aGraph, 'alice', server)
    const { cleanup: cb } = connect(bGraph, 'bob', server)

    aGraph.addNode(node('a'))
    await settle()
    aGraph.reinforceNode('a', 0.2)
    await settle()
    bGraph.reinforceNode('a', 0.3)
    await settle()

    expect(aGraph.getActivation('a')).toBeCloseTo(0.5, 5)
    expect(bGraph.getActivation('a')).toBeCloseTo(0.5, 5)
    ca(); cb()
  })

  it('emits a single activationUpdate op for a reinforcement (no echo)', async () => {
    const server = new SyncServer()
    const aGraph = new PolyGraph()
    const bGraph = new PolyGraph()
    const { cleanup: ca } = connect(aGraph, 'alice', server)
    const { cleanup: cb } = connect(bGraph, 'bob', server)

    aGraph.addNode(node('a'))
    await settle()
    aGraph.reinforceNode('a', 0.2)
    await settle()

    const updates = server.ops.filter(o => o.kind === 'activationUpdate')
    expect(updates).toHaveLength(1)
    expect(updates[0].payload).toMatchObject({ node: 'a', delta: 0.2 })
    expect(bGraph.getActivation('a')).toBeCloseTo(0.2, 5)
    ca(); cb()
  })

  it('drops sub-threshold activation deltas from the op log', async () => {
    const server = new SyncServer()
    const graph = new PolyGraph()
    const { cleanup } = connect(graph, 'alice', server)
    graph.addNode(node('a'))
    await settle()

    graph.reinforceNode('a', 0.01)
    await settle()
    expect(server.ops.filter(o => o.kind === 'activationUpdate')).toHaveLength(0)

    graph.reinforceNode('a', 0.2)
    await settle()
    expect(server.ops.filter(o => o.kind === 'activationUpdate')).toHaveLength(1)
    expect(server.ops.filter(o => o.kind === 'activationUpdate')[0].payload.delta).toBeCloseTo(0.2, 5)
    cleanup()
  })

  it('max-merges durable activation when a full node payload arrives (snapshot catch-up)', async () => {
    const server = new SyncServer()
    const aGraph = new PolyGraph()
    // Alice's addNode op carries her full durable activation total.
    const { cleanup: ca } = connect(aGraph, 'alice', server)
    aGraph.addNode(node('a', 1, { score: 0.6, importance: 0.3, reinforcementCount: 2, lastMeaningfulActivation: Date.now() }))
    await settle()

    const bGraph = new PolyGraph()
    bGraph.addNode(node('a', 1, { score: 0.4, importance: 0.1, reinforcementCount: 1, lastMeaningfulActivation: Date.now() }))
    const { client: bob, cleanup: cb } = connect(bGraph, 'bob', server, false)

    bob.requestSync()
    await settle()

    // Max-merge: bob keeps the stronger total instead of last-write-wins.
    expect(bGraph.getActivation('a')).toBeCloseTo(0.6, 5)
    ca(); cb()
  })

  it('persists activation through a SyncAdapter-backed graph', async () => {
    const { SyncAdapter } = await import('../src/sync/adapter')
    const inner = new MemoryAdapter()
    const graph = new PolyGraph(new SyncAdapter(inner, 'sync-client'))
    graph.addNode(node('a'))
    graph.reinforceNode('a', 0.7)
    await graph.flush()

    const ops = (graph.persistence as unknown as { oplog: { all: unknown[] } }).oplog.all
    expect(ops).toHaveLength(1)
    const payload = (ops[0] as { payload: { activation?: NodeActivation } }).payload
    expect(payload.activation!.score).toBeCloseTo(0.7, 5)
  })
})

/**
 * Example: Multi-agent coordination board
 *
 * Four AI agents — a Planner, a Researcher, a Developer, and a Reviewer —
 * collaborate on a shared software-delivery board. Each agent runs its own
 * PolyGraph replica and stays in sync through one shared SyncServer over
 * in-memory MemoryTransports. Every mutation an agent makes is captured by its
 * SyncClient, relayed by the server, and applied to every other replica, so the
 * "board" converges to the same state everywhere.
 *
 * It shows: real-time sync + echo suppression, snapshot catch-up for late and
 * reconnecting agents, reactive change events (each agent watches the board),
 * agentic retrieval with the built-in feature-hash embedding provider
 * (the Researcher semantically finds relevant reference docs and links them to
 * a task), and edge-ownership cascade replicated across every agent.
 *
 * Run: npx tsx examples/agent-board.ts
 */

import { PolyGraph, MemoryAdapter, FeatureHashEmbedding, cosineSimilarity } from '../src/index'
import { SyncServer, SyncClient, MemoryTransport } from '../src/sync'
import type { GraphChangeEvent } from '../src/index'
import type { SyncServerClient } from '../src/sync'

// ────────────────────────────────────────────────────────────
// HELPERS
// ────────────────────────────────────────────────────────────

/** MemoryTransport delivers asynchronously; settle lets messages propagate. */
const settle = () => new Promise<void>((resolve) => setTimeout(resolve, 20))

function assert(condition: boolean, message: string): void {
  if (!condition) throw new Error(`Assertion failed: ${message}`)
}

/** Deterministic, data-independent snapshot of a replica's full board state. */
function fingerprint(graph: PolyGraph): string {
  const nodes = graph.query().toArray().map(n => n.id).sort().join('|')
  const edges = graph.query().toArray()
    .flatMap(n => graph.getEdges(n.id).map(e => `${n.id} ${e.type} ${e.target}`))
    .sort().join('|')
  return `N[${nodes}] E[${edges}] V=${graph.vectors.size}`
}

/**
 * Pin every replica's server cursor to the current op-log length. A client
 * whose cursor has fallen behind the server triggers a full re-sync on its
 * next incoming delta (which would re-apply its own earlier ops), so we keep
 * all cursors caught up between agent turns.
 */
async function syncAll(agents: Agent[]): Promise<void> {
  await settle()
  for (const a of agents) a.client.requestSync()
  await settle()
}

// ────────────────────────────────────────────────────────────
// AGENT BASE: one local replica + one SyncClient to the board
// ────────────────────────────────────────────────────────────

abstract class Agent {
  readonly name: string
  readonly graph: PolyGraph
  readonly client: SyncClient
  /** Every mutation the agent observes — its own and remote peers'. */
  readonly ledger: GraphChangeEvent[] = []
  private remote: SyncServerClient
  private serverT: MemoryTransport

  constructor(name: string, server: SyncServer) {
    this.name = name
    this.graph = new PolyGraph(new MemoryAdapter(), undefined, new FeatureHashEmbedding())
    this.graph.changes.subscribe(event => this.ledger.push(event))

    // agentT is this agent's leg; serverT is the server's leg of the same link.
    const [agentT, serverT] = MemoryTransport.pair()
    this.serverT = serverT
    this.remote = { send: (msg) => agentT.onMessage?.(msg), clientId: this.name }
    serverT.onMessage = server.addClient(this.remote)
    this.client = new SyncClient({ graph: this.graph, transport: agentT, clientId: this.name, autoFlush: true, retryMs: 60 })
    // Pull whatever already happened on the board (snapshot catch-up).
    this.client.requestSync()
  }

  /** Drop the link: the agent's SyncClient keeps capturing local edits and retrying. */
  simulateOffline(server: SyncServer): void {
    server.removeClient(this.remote)
    this.serverT.onMessage = null
  }

  /** Register a fresh transport, resend pending ops, and catch up on missed ops. */
  reconnect(server: SyncServer): void {
    server.removeClient(this.remote)
    const [agentT, serverT] = MemoryTransport.pair()
    this.serverT = serverT
    this.remote = { send: (msg) => agentT.onMessage?.(msg), clientId: this.name }
    serverT.onMessage = server.addClient(this.remote)
    this.client.reconnect(agentT)
  }

  seenCount(type: GraphChangeEvent['type'], nodeId?: string): number {
    return this.ledger.filter(e => e.type === type && (nodeId === undefined || e.nodeId === nodeId)).length
  }
}

// ────────────────────────────────────────────────────────────
// 1. PLANNER — decompose the feature request into a task DAG
// ────────────────────────────────────────────────────────────

class PlannerAgent extends Agent {
  readonly taskIds = ['task:t1', 'task:t2', 'task:t3', 'task:t4']

  async seed(): Promise<void> {
    const knowledge: { id: string; title: string; text: string }[] = [
      { id: 'know:dark', title: 'Dark Mode Design Guide', text: 'dark mode ui color palette contrast accessibility' },
      { id: 'know:wcag', title: 'WCAG Contrast Rules', text: 'wcag color contrast readability accessibility guidelines' },
      { id: 'know:tokens', title: 'Design Token System', text: 'css design tokens theme variables theming' },
      { id: 'know:layout', title: 'Responsive Layout Patterns', text: 'responsive layout grid flexbox breakpoints' },
      { id: 'know:security', title: 'Auth Security Notes', text: 'authentication authorization session management' },
    ]
    for (const k of knowledge) {
      await this.graph.addNodeWithEmbedding({
        id: k.id, type: 'knowledge', data: { title: k.title },
        insertedAt: Date.now(), updatedAt: Date.now(),
      }, k.text)
    }
    for (const [id, name] of [['agent:planner', 'planner'], ['agent:researcher', 'researcher'], ['agent:dev', 'dev'], ['agent:reviewer', 'reviewer']] as const) {
      this.graph.addNode({ id, type: 'agent', data: { name }, insertedAt: Date.now(), updatedAt: Date.now() })
    }
    this.graph.addNode({
      id: 'fr:dark-mode', type: 'feature', data: { title: 'Add dark mode to the web app', status: 'open' },
      insertedAt: Date.now(), updatedAt: Date.now(),
    })
  }

  decompose(): void {
    const tasks: Record<string, { title: string; deps: string[] }> = {
      'task:t1': { title: 'Write design spec', deps: [] },
      'task:t2': { title: 'Add theme token palette', deps: ['task:t1'] },
      'task:t3': { title: 'Implement dark mode components', deps: ['task:t2'] },
      'task:t4': { title: 'Add contrast accessibility tests', deps: ['task:t2', 'task:t3'] },
    }
    for (const [id, t] of Object.entries(tasks)) {
      this.graph.addNode({
        id, type: 'task', data: { title: t.title, status: 'pending', owner: 'dev' },
        insertedAt: Date.now(), updatedAt: Date.now(),
      })
      this.graph.addEdge(id, 'PART_OF', 'fr:dark-mode')
      this.graph.addEdge(id, 'ASSIGNED_TO', 'agent:dev')
    }
    for (const [id, t] of Object.entries(tasks)) {
      for (const dep of t.deps) {
        this.graph.addEdge(id, 'DEPENDS_ON', dep)
      }
    }
  }
}

// ────────────────────────────────────────────────────────────
// 2. RESEARCHER — retrieve relevant reference docs (semantic memory)
// ────────────────────────────────────────────────────────────

class ResearcherAgent extends Agent {
  async run(): Promise<string[]> {
    const query = 'dark mode color contrast tokens theme'
    const queryVec = [...(await this.graph.embed(query))]
    const hits = (await this.graph.queryText(query, 0.05, 3)).toArray()
    const byScore = hits
      .map(n => ({ id: n.id, title: (n.data as any).title, score: cosineSimilarity(queryVec, [...n.vector!]) }))
      .sort((a, b) => b.score - a.score)

    for (const h of byScore) {
      this.graph.addEdge('task:t1', 'MENTIONS', h.id)
    }
    return byScore.map(h => `${h.id} ${h.title} (score=${h.score.toFixed(3)})`)
  }
}

// ────────────────────────────────────────────────────────────
// 3. DEVELOPER — execute ready tasks, produce artifacts, fix reviews
// ────────────────────────────────────────────────────────────

const ARTIFACTS: Record<string, { file: string; content: string }> = {
  'task:t1': { file: 'spec/dark-mode.md', content: 'design spec dark mode color palette contrast tokens' },
  'task:t2': { file: 'styles/theme-tokens.css', content: 'theme token palette css custom properties dark mode' },
  'task:t3': { file: 'components/dark-mode.tsx', content: 'dark mode components theme hook styles' },
  'task:t4': { file: 'tests/contrast.test.ts', content: 'contrast accessibility tests wcag color check' },
}

class DeveloperAgent extends Agent {
  async run(): Promise<string> {
    const log: string[] = []
    for (let pass = 0; pass < 10; pass++) {
      const fixed = this.fixRequestedChanges(log)
      const task = this.nextReadyTask()
      if (!task && !fixed) break
      if (task) await this.execute(task, log)
    }
    return log.join('\n')
  }

  private fixRequestedChanges(log: string[]): boolean {
    let fixed = false
    for (const taskId of this.graph.query().whereNodeType('task').ids()) {
      const task = this.graph.getNode(taskId)
      if ((task?.data as any)?.status !== 'done') continue
      const reviews = this.graph.getEdgeTargets(taskId, 'HAS_REVIEW')
      const open = reviews.find(r => (this.graph.getNode(r)?.data as any)?.verdict === 'changes-requested')
      const alreadyAddressed = reviews.some(r => (this.graph.getNode(r)?.data as any)?.addressed === true)
      if (!open || alreadyAddressed) continue
      const artifact = this.graph.getEdgeTargets(taskId, 'PRODUCED')[0]
      const revision = ((this.graph.getNode(artifact)?.data as any)?.revision ?? 1) + 1
      if (artifact) this.graph.updateNode(artifact, { revision })
      this.setStatus(taskId, 'in_progress')
      this.setStatus(taskId, 'done')
      // Mark the review addressed so the fix runs exactly once per review.
      this.graph.updateNode(open, { ...this.graph.getNode(open)!.data, addressed: true })
      log.push(`    ${taskId} → ${artifact} revision ${revision} → done (fix applied)`)
      fixed = true
    }
    return fixed
  }

  private nextReadyTask(): string | null {
    const tasks = this.graph.query().whereNodeType('task').toArray()
    for (const t of tasks.sort((a, b) => a.id.localeCompare(b.id))) {
      if ((t.data as any).status === 'done') continue
      const deps = this.graph.getEdgeTargets(t.id, 'DEPENDS_ON')
      const ready = deps.every(dep => (this.graph.getNode(dep)?.data as any)?.status === 'done')
      if (ready) return t.id
    }
    return null
  }

  private async execute(taskId: string, log: string[]): Promise<void> {
    const meta = ARTIFACTS[taskId]
    this.setStatus(taskId, 'in_progress')
    const artifactId = `artifact:${taskId.slice('task:'.length)}`
    await this.graph.addNodeWithEmbedding({
      id: artifactId, type: 'artifact',
      data: { file: meta.file, title: taskId, revision: 1 },
      insertedAt: Date.now(), updatedAt: Date.now(),
    }, meta.content)
    this.graph.addEdge(taskId, 'PRODUCED', artifactId, {}, 'owned')
    this.setStatus(taskId, 'done')
    log.push(`    ${taskId} → ${artifactId} (${meta.file}) → done`)
  }

  private setStatus(taskId: string, status: string): void {
    const task = this.graph.getNode(taskId)
    if (!task) return
    this.graph.updateNode(taskId, { ...task.data, status })
  }
}

// ────────────────────────────────────────────────────────────
// 4. REVIEWER — review done tasks; t1 gets one requested change
// ────────────────────────────────────────────────────────────

class ReviewerAgent extends Agent {
  run(): string {
    const log: string[] = []
    for (const taskId of this.graph.query().whereNodeType('task').ids().sort()) {
      const task = this.graph.getNode(taskId)
      if ((task?.data as any)?.status !== 'done') continue
      const reviews = this.graph.getEdgeTargets(taskId, 'HAS_REVIEW').map(r => this.graph.getNode(r))
      if (reviews.some(r => (r?.data as any)?.verdict === 'approved')) continue
      const requested = reviews.some(r => (r?.data as any)?.verdict === 'changes-requested')
      const id = `review:${taskId}:${reviews.length + 1}`
      const firstReviewOfT1 = taskId === 'task:t1' && reviews.length === 0
      const verdict = requested ? 'approved' : firstReviewOfT1 ? 'changes-requested' : 'approved'
      const issue = firstReviewOfT1 ? 'missing focus-visible indicator tokens' : undefined
      this.graph.addNode({
        id, type: 'review', data: { verdict, issue }, insertedAt: Date.now(), updatedAt: Date.now(),
      })
      this.graph.addEdge(taskId, 'HAS_REVIEW', id)
      this.graph.addEdge(id, 'AUTHORED_BY', 'agent:reviewer')
      log.push(`    ${taskId} → ${verdict}${issue ? `: ${issue}` : ''}`)
    }
    return log.join('\n')
  }
}

// ────────────────────────────────────────────────────────────
// RUN THE SCENARIO
// ────────────────────────────────────────────────────────────

console.log('── Multi-agent coordination board ──')
console.log('  Planner, Researcher, Developer + Reviewer share one board graph,')
console.log('  each with its own replica synchronized through a single SyncServer.\n')

const server = new SyncServer()
const planner = new PlannerAgent('planner', server)

// ── BOOTSTRAP: seed knowledge + identities on the planner ──
await planner.seed()
await settle()
console.log('── BOOTSTRAP ──')
console.log(`  Seeded 5 knowledge docs, 4 agents, feature request 'Add dark mode to the web app' on Planner`)

// Late-joining agents catch up via snapshot.
const researcher = new ResearcherAgent('researcher', server)
const developer = new DeveloperAgent('dev', server)
const reviewer = new ReviewerAgent('reviewer', server)

const agents = [planner, researcher, developer, reviewer]
// Pin every replica's cursor to the seeded op-log before any peer activity, so
// incoming deltas never look like a gap (which would force a full re-sync).
await syncAll(agents)

const seededFp = fingerprint(planner.graph)
for (const a of agents) assert(fingerprint(a.graph) === seededFp, `${a.name} caught up to the seeded board`)
console.log(`  All ${agents.length} replicas caught up via snapshot: ${planner.graph.size} nodes, ${planner.graph.vectors.size} vectors\n`)

// ── 1. PLANNER decomposes the feature into a task DAG ──
planner.decompose()
await syncAll(agents)
console.log('── 1. PLANNER: decompose feature request ──')
const dag: string[] = []
for (const id of planner.taskIds) {
  const deps = planner.graph.getEdgeTargets(id, 'DEPENDS_ON')
  dag.push(`    ${id.padEnd(10)} ${(planner.graph.getNode(id)?.data as any)?.title.padEnd(30)} deps=[${deps.join(', ') || '—'}]`)
}
console.log(dag.join('\n'))
console.log('  Planner added 4 task nodes + 12 edges; every replica now sees the same DAG\n')

// ── 2. RESEARCHER retrieves relevant docs and links them to the spec task ──
const retrieved = await researcher.run()
await syncAll(agents)
console.log('── 2. RESEARCHER: semantic retrieval ──')
console.log('  queryText("dark mode color contrast tokens theme") →')
for (const line of retrieved) console.log(`    ${line}`)
console.log('  Researcher linked retrieved docs to task:t1 via MENTIONS edges\n')

// ── 3. DEVELOPER executes the task DAG in dependency order ──
const devLog = await developer.run()
await syncAll(agents)
console.log('── 3. DEVELOPER: execute the task DAG ──')
console.log(devLog)
console.log('  (each task owns its artifact via a PRODUCED edge)\n')

// ── 4. REVIEWER reviews ──
const reviewLog = reviewer.run()
await syncAll(agents)
console.log('── 4. REVIEWER: review done tasks ──')
console.log(reviewLog + '\n')

// ── 5. DEVELOPER fixes the requested change ──
const fixLog = await developer.run()
await syncAll(agents)
console.log('── 5. DEVELOPER: address review ──')
console.log(fixLog + '\n')

// ── 6. REVIEWER re-reviews ──
const reReviewLog = reviewer.run()
await syncAll(agents)
console.log('── 6. REVIEWER: re-review ──')
console.log(reReviewLog + '\n')

// ── CONVERGENCE: every replica must be byte-identical ──
console.log('── CONVERGENCE ──')
const finalFp = fingerprint(planner.graph)
for (const a of agents) assert(fingerprint(a.graph) === finalFp, `${a.name} converged with the board`)
assert(finalFp.includes('V=9'), '9 vectors (5 knowledge + 4 artifacts) synced to every replica')
const done = planner.graph.query().whereNodeType('task').toArray().filter(t => (t.data as any).status === 'done').length
assert(done === 4, 'all 4 tasks delivered')
const reviews = planner.graph.query().whereNodeType('review').toArray()
assert(reviews.length === 5, '5 reviews recorded (t1 reviewed twice)')
assert(reviews.every(r => ['approved', 'changes-requested'].includes((r.data as any).verdict)), 'every review has a verdict')
assert(reviews.some(r => r.id === 'review:task:t1:2' && (r.data as any).verdict === 'approved'), 't1 approved after the requested change')
console.log(`  ${finalFp}`)
console.log('  All replicas hold the identical board state; task:t1 was approved after one fix cycle ✓\n')

// ── ECHO SUPPRESSION: sync settles with no loopback or echo storm ──
// The relay never broadcasts an op back to its sender, and catch-ups skip a
// client's own ops, so every mutation is observed exactly once per agent.
console.log('── ECHO SUPPRESSION ──')
for (const a of agents) {
  assert(a.seenCount('node_added', 'task:t1') === 1, `${a.name} saw task:t1 exactly once`)
  assert(a.seenCount('node_added', 'artifact:t1') === 1, `${a.name} saw artifact:t1 exactly once`)
  assert(a.client.pendingOps.length === 0, `${a.name} acked every op`)
}
const t1Adds = server.ops.filter(o => o.kind === 'addNode' && (o.payload as any).id === 'task:t1').length
assert(t1Adds === 1, 'task:t1 reaches the relay exactly once')
console.log(`  Every agent observed task:t1 and artifact:t1 exactly once — no loopback, no replay`)
console.log(`  Relay holds exactly ${t1Adds} copy of task:t1; every agent acked all ops\n`)

// ── OFFLINE / RECONNECT: a dropped agent retries, then catches up both ways ──
console.log('── OFFLINE / RECONNECT ──')
reviewer.simulateOffline(server)
console.log('  reviewer goes offline (removed from the relay)')

planner.graph.addNode({
  id: 'note:release', type: 'note', data: { text: 'Ship dark mode in v2.5' },
  insertedAt: Date.now(), updatedAt: Date.now(),
})
planner.graph.addEdge('note:release', 'PART_OF', 'fr:dark-mode')
const reviewNode = planner.graph.getNode('review:task:t1:1')!
await settle()
assert(!reviewer.graph.getNode('note:release'), 'reviewer did not see the note while offline')

reviewer.graph.updateNode(reviewNode.id, { ...reviewNode.data, issue: `${reviewNode.data.issue} (addressed in revision 2)` })
await settle()
assert(reviewer.client.pendingOps.length > 0, 'reviewer retained unacknowledged ops locally')

reviewer.reconnect(server)
await syncAll(agents)

const reconvergedFp = fingerprint(planner.graph)
for (const a of agents) assert(fingerprint(a.graph) === reconvergedFp, `${a.name} reconverged after reconnect`)
assert(reviewer.graph.getNode('note:release') !== undefined, 'reviewer caught up on the missed note')
assert((planner.graph.getNode(reviewNode.id)?.data as any).issue.includes('revision 2'), 'offline edit delivered to the board')
assert(reviewer.client.pendingOps.length === 0, 'reviewer acked everything after reconnect')
const reviewUpdates = server.ops.filter(o =>
  o.kind === 'updateNode' && (o.payload as any).id === reviewNode.id && ((o.payload as any).data as any)?.issue?.includes('revision 2'),
).length
assert(reviewUpdates === 1, 'offline edit reached the relay exactly once despite retries')
console.log('  reviewer reconnects → resends pending ops + requests a delta for missed ones')
console.log('  Both the offline edit (revision 2) and the missed note are on every replica ✓')
console.log(`  The relay stored the retried offline edit exactly once (deduped by client+seq)`)
console.log(`  ${reconvergedFp}\n`)

// ── OWNERSHIP CASCADE ACROSS REPLICAS ──
// Edge ownership rides inside the edge payload, so it survives replication:
// deleting an owner on one agent cascades to its owned targets everywhere.
console.log('── OWNERSHIP CASCADE ACROSS REPLICAS ──')
planner.graph.addNode({
  id: 'demo:doc', type: 'demo', data: { title: 'owned parent' },
  insertedAt: Date.now(), updatedAt: Date.now(),
})
planner.graph.addNode({
  id: 'demo:child', type: 'demo', data: { title: 'owned child' },
  insertedAt: Date.now(), updatedAt: Date.now(),
})
planner.graph.addEdge('demo:doc', 'OWNS', 'demo:child', {}, 'owned')
await syncAll(agents)

for (const a of agents) {
  const edges = a.graph.getEdges('demo:doc', 'OWNS')
  assert(edges.length === 1 && edges[0].data?.__ownership === 'owned', `${a.name} replicated the owned edge`)
}
console.log('  Planner created demo:doc ──[OWNS, owned]──> demo:child')

planner.graph.removeNode('demo:doc')
await syncAll(agents)

for (const a of agents) {
  assert(a.graph.getNode('demo:doc') === undefined, `${a.name} cascaded demo:doc`)
  assert(a.graph.getNode('demo:child') === undefined, `${a.name} cascaded demo:child`)
}
console.log('  Deleting demo:doc on Planner cascaded to demo:child on every replica ✓\n')

console.log('✓ Multi-agent coordination board example completed')

/**
 * Example: Adaptive memory with the activation model
 *
 * A Nostr-style client that stops being a passive data store and starts
 * maintaining an adaptive memory. The `ActivationEngine` turns the graph into
 * a working-memory system:
 *
 *   - reinforcement: reading / opening / searching a node raises its durable
 *     activation (which persists and, with the sync layer, replicates additively)
 *   - semantic pulse: a query scores the region around itself (vector matches +
 *     recency + usage), and `absorb` reinforces the relevant nodes
 *   - spreading activation: reading an article warms its related neighbours
 *   - attention promotion: small local signals accumulate and only become
 *     durable (and syncable) once they cross a meaningful threshold
 *   - query integration: activation composes with the regular query builder
 *   - decay: activation fades over time as a pure function of elapsed time
 *   - working memory: the "mental state" is just the most-active loaded nodes
 *
 * Run: npx tsx examples/activation.ts
 */

import { PolyGraph, MemoryAdapter, FeatureHashEmbedding, ActivationEngine } from '../src/index'

// ────────────────────────────────────────────────────────────
// SETUP: a small knowledge graph of interests, authors, topics
// ────────────────────────────────────────────────────────────

const graph = new PolyGraph(new MemoryAdapter(), undefined, new FeatureHashEmbedding({ dimensions: 256 }))
const engine = new ActivationEngine(graph)

const add = async (id: string, type: string, text: string, data: Record<string, unknown> = {}) => {
  await graph.addNodeWithEmbedding(
    { id, type, data: { text, ...data }, insertedAt: Date.now(), updatedAt: Date.now() },
    text,
  )
}

// The graph's durable knowledge: topics and the people you follow.
await add('topic:ai', 'topic', 'artificial intelligence local models machine learning')
await add('topic:vector-db', 'topic', 'vector databases embedding search similarity')
await add('topic:nostr', 'topic', 'nostr decentralized social networking notes')
await add('topic:sports', 'topic', 'sports games matches scores')

await add('author:alice', 'author', 'alice writes about local machine learning and vector search')
await add('author:bob', 'author', 'bob writes about running and sports')

graph.addEdge('author:alice', 'FOLLOWS', 'topic:ai')
graph.addEdge('author:alice', 'FOLLOWS', 'topic:vector-db')
graph.addEdge('author:bob', 'FOLLOWS', 'topic:sports')

// Your long-term interests sit at the hub; everything connects through them.
graph.addEdge('topic:ai', 'RELATES', 'topic:vector-db')
graph.addEdge('topic:vector-db', 'RELATES', 'topic:nostr')

console.log('── Activation model: Nostr client ──')
console.log(`  Seeded ${graph.size} nodes (topics, authors) + ${graph.vectors.size} vectors\n`)

// ────────────────────────────────────────────────────────────
// 1. A new event arrives — relevant to your interests
// ────────────────────────────────────────────────────────────

await add('event:123', 'event', 'New developments in local AI models for on-device vector search')
graph.addEdge('event:123', 'AUTHORED_BY', 'author:alice')
graph.addEdge('event:123', 'MENTIONS', 'topic:ai')
graph.addEdge('event:123', 'MENTIONS', 'topic:vector-db')

// A second event — unrelated to your world model.
await add('event:456', 'event', 'Random discussion about last night sports game')
graph.addEdge('event:456', 'AUTHORED_BY', 'author:bob')
graph.addEdge('event:456', 'MENTIONS', 'topic:sports')

console.log('── 1. Ingesting two events ──')
console.log('  event:123  "New developments in local AI models"  (from Alice, mentions AI/vector-db)')
console.log('  event:456  "Random discussion about last night sports game"  (from Bob, mentions sports)\n')

// ────────────────────────────────────────────────────────────
// 2. Semantic pulse: why should this data be active right now?
// ────────────────────────────────────────────────────────────

const pulse = await engine.pulse('local AI model vector search')
const hot123 = pulse.get('event:123') ?? 0
const hot456 = pulse.get('event:456') ?? 0

console.log('── 2. Semantic pulse: "local AI model vector search" ──')
console.log(`  event:123  composite = ${hot123.toFixed(2)}  → becomes part of the active graph`)
console.log(`  event:456  composite = ${hot456.toFixed(2)}  → stays cold`)
console.log(`  (composite = semantic hit + spreading activation + recency + usage)`)

// Absorb: the relevant region gets durably reinforced (persisted + synced).
await engine.absorb('local AI model vector search')
console.log('  absorb() reinforced event:123 and its topic neighbourhood\n')

// ────────────────────────────────────────────────────────────
// 3. Spreading activation: reading warms the surrounding graph
// ────────────────────────────────────────────────────────────

console.log('── 3. Spreading activation from a Polypack-article read ──')
await add('article:polypack', 'article', 'polypack property graph vector search engine')
await add('article:qdrant', 'article', 'qdrant vector database open source')
graph.addEdge('article:polypack', 'RELATES', 'topic:vector-db')
graph.addEdge('article:qdrant', 'RELATES', 'topic:vector-db')

graph.reinforceNode('article:polypack', 1.0, 'user_read')
const spread = engine.spread(['article:polypack'], { depth: 2 })
for (const [id, contribution] of [...spread.entries()].sort((a, b) => b[1] - a[1])) {
  console.log(`  ${id.padEnd(22)} +${contribution.toFixed(2)}`)
}

// ────────────────────────────────────────────────────────────
// 4. Working memory: the current "mental state"
// ────────────────────────────────────────────────────────────

console.log('\n── 4. Working memory (top activated nodes) ──')
for (const wm of engine.workingMemory(5)) {
  const node = graph.getNode(wm.id)
  const kind = node?.type
  const text = (node?.data as any)?.text ?? ''
  console.log(`  ${wm.id.padEnd(22)} ${engine.effective(wm.id).toFixed(2)}  [${kind}] ${text.slice(0, 48)}`)
}

console.log('\n  Activate a node to see its score rise, then decay it:')
console.log(`    graph.reinforceNode(id, amount, 'reason')   — durable, synced`)
console.log(`    engine.bumpAttention(id, amount)            — local only, never synced`)
console.log(`    engine.spread(seeds, { depth, decay })      — relational activation`)
console.log(`    engine.pulse(text | vector)                 — semantic region scoring`)
console.log(`    engine.workingMemory(limit)                 — current mental state\n`)

// ────────────────────────────────────────────────────────────
// 5. Attention promotion: small local signals, promoted once meaningful
// ────────────────────────────────────────────────────────────

console.log('── 5. Transient attention, promoted past the threshold ──')
await add('event:789', 'event', 'Another local AI model release')

// A couple of small, sub-threshold signals (e.g. the user glanced at the
// card). These never touch durable state or the sync layer.
engine.bumpAttention('event:789', 0.02)
console.log(`  bumpAttention(0.02) → local attention = ${engine.attentionOf('event:789').toFixed(2)}, durable = ${(graph.getActivationState('event:789')?.score ?? 0).toFixed(2)}`)

engine.bumpAttention('event:789', 0.02)
console.log(`  bumpAttention(0.02) → local attention = ${engine.attentionOf('event:789').toFixed(2)}, durable = ${(graph.getActivationState('event:789')?.score ?? 0).toFixed(2)}`)

// This delta pushes the accumulated total past minReinforceDelta (0.05):
// it's promoted into durable reinforcement and local attention resets to 0.
engine.bumpAttention('event:789', 0.03)
console.log(`  bumpAttention(0.03) → local attention = ${engine.attentionOf('event:789').toFixed(2)}, durable = ${(graph.getActivationState('event:789')?.score ?? 0).toFixed(2)}  (promoted!)`)
console.log('  Small events stayed local; only the accumulated, meaningful total became durable + syncable.\n')

// ────────────────────────────────────────────────────────────
// 6. Query integration: activation as a first-class filter/sort
// ────────────────────────────────────────────────────────────

console.log('── 6. Querying by activation ──')
const feed = graph.query().whereActivated(0.05).orderByActivation('desc').toArray()
console.log('  query().whereActivated(0.05).orderByActivation("desc")')
for (const n of feed) {
  console.log(`    ${n.id.padEnd(22)} ${engine.effective(n.id).toFixed(2)}  [${n.type}]`)
}
console.log('  Same shape as any other query — activation composes with whereNodeType, traverse, limit, etc.\n')

// ────────────────────────────────────────────────────────────
// 7. Decay: nothing stays hot forever
// ────────────────────────────────────────────────────────────

const before = engine.effective('event:123')
// Simulate a day passing on the score curve.
const now = Date.now()
const state = graph.getActivationState('event:123')!
const decayed = state.score * Math.pow(0.5, (now - state.lastMeaningfulActivation) / (24 * 3_600_000))
console.log('── 7. Decay is a pure function of elapsed time ──')
console.log(`  event:123 now ${before.toFixed(2)}; after one 24h half-life → ~${decayed.toFixed(2)}`)
console.log('  It can be reinforced again at any time.\n')

engine.dispose()
console.log('✓ Activation example completed')

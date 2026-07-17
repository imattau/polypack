/**
 * polypack — comprehensive example
 *
 * A property graph library with vector similarity search, edge ownership
 * semantics, reactive change events, and pluggable persistence.
 *
 * Run: npx tsx examples/readme.ts
 */

import {
  PolyGraph,
  VectorIndex,
  MemoryAdapter,
  IndexedDBAdapter,
  cosineSimilarity,
  euclideanSimilarity,
} from '../src/index'
import type { PolyNode, EdgeOwnership, GraphChangeEvent, DistanceFunction } from '../src/index'

// ────────────────────────────────────────────────────────────
// 1. CREATE A GRAPH
// ────────────────────────────────────────────────────────────
//
// PolyGraph with no adapter = in-memory only (ephemeral).
// Pass a MemoryAdapter for testable persistence, or
// IndexedDBAdapter for browser persistence.

const graph = new PolyGraph(new MemoryAdapter())
console.log('─ PolyGraph created (MemoryAdapter)')

// ────────────────────────────────────────────────────────────
// 2. ADD NODES WITH VECTORS
// ────────────────────────────────────────────────────────────
//
// Nodes are polymorphic — type is any string. Vectors are
// Float64Array and are auto-registered in the VectorIndex.

const docs = [
  { id: 'doc_1', type: 'document', data: { title: 'Quantum Computing', category: 'science' }, vec: [0.95, 0.20, 0.10, 0.05] },
  { id: 'doc_2', type: 'document', data: { title: 'Deep Learning Basics', category: 'science' }, vec: [0.85, 0.40, 0.15, 0.10] },
  { id: 'doc_3', type: 'document', data: { title: 'Cooking Pasta', category: 'food' }, vec: [0.10, 0.25, 0.90, 0.05] },
  { id: 'doc_4', type: 'document', data: { title: 'Italian Sauces', category: 'food' }, vec: [0.15, 0.20, 0.85, 0.10] },
  { id: 'doc_5', type: 'document', data: { title: 'Rust vs Go', category: 'tech' }, vec: [0.70, 0.60, 0.05, 0.20] },
  { id: 'alice', type: 'user', data: { name: 'Alice', interests: 'science' }, vec: [0.90, 0.30, 0.10, 0.08] },
  { id: 'bob', type: 'user', data: { name: 'Bob', interests: 'food' }, vec: [0.10, 0.20, 0.92, 0.05] },
]

for (const d of docs) {
  graph.addNode({
    id: d.id,
    type: d.type,
    data: d.data,
    vector: new Float64Array(d.vec),
    insertedAt: Date.now(),
    updatedAt: Date.now(),
  })
}
console.log(`  Added ${graph.size} nodes (${graph.vectors.size} vectors registered)`)

// ────────────────────────────────────────────────────────────
// 3. ADD EDGES WITH OWNERSHIP
// ────────────────────────────────────────────────────────────
//
// Ownership semantics:
//   owned     — target deleted when source is deleted
//   shared    — target survives if other sources remain
//   reference — just a link, no lifecycle effect

graph.addEdge('alice', 'AUTHORED', 'doc_1', { weight: 1 }, 'owned')
graph.addEdge('alice', 'AUTHORED', 'doc_2', { weight: 1 }, 'owned')
graph.addEdge('bob', 'AUTHORED', 'doc_3', { weight: 1 }, 'owned')
graph.addEdge('bob', 'AUTHORED', 'doc_4', { weight: 1 }, 'owned')
graph.addEdge('alice', 'AUTHORED', 'doc_5', { weight: 1 }, 'owned')

graph.addEdge('doc_1', 'REFERENCES', 'doc_2')      // default = reference
graph.addEdge('doc_2', 'REFERENCES', 'doc_5')
graph.addEdge('doc_3', 'REFERENCES', 'doc_4')

graph.addEdge('alice', 'FOLLOWS', 'bob', undefined, 'shared')

console.log('  Added edges with owned / shared / reference semantics')

// ────────────────────────────────────────────────────────────
// 4. QUERY: TYPE + ATTRIBUTE + SORT + LIMIT
// ────────────────────────────────────────────────────────────

const scienceDocs = graph.query()
  .whereNodeType('document')
  .where('category', 'science')
  .orderBy('title', 'asc')
  .toArray()
console.log(`\n  Science docs: ${scienceDocs.map(n => n.data.title).join(', ')}`)

// ────────────────────────────────────────────────────────────
// 5. VECTOR SIMILARITY SEARCH
// ────────────────────────────────────────────────────────────

// Query: "find documents related to machine learning"
const queryVec = [0.85, 0.45, 0.10, 0.08]

// Standalone vector search (returns scored IDs)
const raw = graph.vectors.query(queryVec, 3, 0.5)
console.log(`\n  Vector search (top 3, threshold 0.5):`)
for (const r of raw) {
  const node = graph.getNode(r.id)
  console.log(`    ${r.id.padEnd(8)} score=${r.score.toFixed(3)}  ${node ? (node.data as any).title || (node.data as any).name : '?'}`)
}

// Combined with graph filters — "only documents, sorted by similarity"
const similarDocs = graph.query()
  .whereNodeType('document')
  .similarTo(queryVec, 0.5, 2)
  .toArray()
console.log(`\n  GraphQuery with similarTo (type=document, top=2):`)
for (const n of similarDocs) console.log(`    ${(n.data as any).title} (id=${n.id})`)

// ────────────────────────────────────────────────────────────
// 6. BFS TRAVERSAL
// ────────────────────────────────────────────────────────────

// Walk the REFERENCES chain starting from doc_1
const refChain = graph.query()
  .where('title', 'Quantum Computing')
  .traverse('REFERENCES', 3, 'out')
  .toArray()
console.log(`\n  REFERENCES chain from doc_1 (BFS, depth 3):`)
for (const n of refChain) console.log(`    ${(n.data as any).title || n.id}`)

// ────────────────────────────────────────────────────────────
// 7. CHANGE EVENTS (RxJS)
// ────────────────────────────────────────────────────────────

const events: GraphChangeEvent[] = []
const sub = graph.changes.subscribe(e => events.push(e))

graph.addNode({ id: 'temp', type: 'draft', data: {}, insertedAt: Date.now(), updatedAt: Date.now() })
graph.addEdge('temp', 'DRAFT_OF', 'doc_1')
graph.removeNode('temp')

sub.unsubscribe()
console.log(`\n  Change events captured: ${events.length}`)
console.log(`    Types: ${[...new Set(events.map(e => e.type))].join(', ')}`)

// ────────────────────────────────────────────────────────────
// 8. BATCHING (coalesces change events)
// ────────────────────────────────────────────────────────────

const batchedEvents: GraphChangeEvent[] = []
const sub2 = graph.changes.subscribe(e => batchedEvents.push(e))

graph.startBatch()
for (let i = 0; i < 5; i++) {
  graph.addNode({ id: `batch_${i}`, type: 'batch', data: {}, insertedAt: Date.now(), updatedAt: Date.now() })
}
expect(batchedEvents.length === 0) // nothing emitted yet
graph.endBatch()
expect(batchedEvents.length === 5) // all flushed at once

sub2.unsubscribe()
console.log(`  Batch insert: ${batchedEvents.length} events coalesced into one flush`)

// ────────────────────────────────────────────────────────────
// 9. CASCADING DELETION
// ────────────────────────────────────────────────────────────

// alice owns doc_1, doc_2, doc_5. Deleting alice cascades:
graph.removeNode('alice')
console.log(`\n  After cascading delete of 'alice':`)
console.log(`    alice exists: ${!!graph.getNode('alice')}`)
console.log(`    doc_1 exists: ${!!graph.getNode('doc_1')}  (was owned by alice)`)
console.log(`    doc_2 exists: ${!!graph.getNode('doc_2')}  (was owned by alice)`)
console.log(`    doc_5 exists: ${!!graph.getNode('doc_5')}  (was owned by alice)`)
console.log(`    bob exists:   ${!!graph.getNode('bob')}    (shared edge only — not owned)`)

// ────────────────────────────────────────────────────────────
// 10. ORPHAN DETECTION
// ────────────────────────────────────────────────────────────

const orphans: string[] = []
class OrphanAwareGraph extends PolyGraph {
  protected onOrphan(id: string): void { orphans.push(id) }
}

const og = new OrphanAwareGraph(new MemoryAdapter())
og.addNode({ id: 'src', type: 't', data: {}, insertedAt: 1, updatedAt: 1 })
og.addNode({ id: 'tgt', type: 't', data: {}, insertedAt: 2, updatedAt: 2 })
og.addEdge('src', 'LINKS', 'tgt', undefined, 'shared')

og.removeEdges('src', 'LINKS', 'tgt')
console.log(`\n  Orphan hook fired for: ${orphans.join(', ')}`)

// ────────────────────────────────────────────────────────────
// 11. CUSTOM DISTANCE FUNCTION
// ────────────────────────────────────────────────────────────

const hammingLike: DistanceFunction = (a, b) => {
  let same = 0
  for (let i = 0; i < a.length; i++) {
    if (Math.round(a[i]) === Math.round(b[i])) same++
  }
  return same / a.length
}

const customIdx = new VectorIndex(undefined, hammingLike)
customIdx.add('a', [0.9, 0.1, 0.8, 0.2])
customIdx.add('b', [0.1, 0.9, 0.1, 0.9])
const similar = customIdx.query([0.95, 0.05, 0.85, 0.15], 1)
console.log(`\n  Custom distance (hamming-like) search: top = ${similar[0].id}`)

// ────────────────────────────────────────────────────────────
// 12. PERSISTENCE ROUND-TRIP
// ────────────────────────────────────────────────────────────

const adapter = new MemoryAdapter()
const g1 = new PolyGraph(adapter)
g1.addNode({ id: 'persisted', type: 'doc', data: { text: 'hello' }, vector: new Float64Array([1, 0, 0]), insertedAt: 1, updatedAt: 1 })
g1.addEdge('persisted', 'TAG', 'tag1')
await g1.save()
console.log(`\n  Saved ${g1.size} node(s) and ${g1.getEdges('persisted').length} edge(s)`)

const g2 = new PolyGraph(adapter)
await g2.warm()
console.log(`  Warmed from persistence: ${g2.size} node(s), ${g2.vectors.size} vector(s)`)
console.log(`  Node text: ${(g2.getNode('persisted')?.data as any).text}`)
console.log(`  Edge targets: ${g2.getEdgeTargets('persisted', 'TAG').join(', ')}`)

// ────────────────────────────────────────────────────────────
// HELPERS
// ────────────────────────────────────────────────────────────

function expect(condition: boolean) {
  if (!condition) throw new Error('Assertion failed')
}

console.log('\n✓ All examples completed successfully')

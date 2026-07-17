/**
 * Example: Personal knowledge base with semantic search
 *
 * Models a research wiki where documents contain sections which contain
 * chunks (paragraphs). Each chunk has an embedding vector for semantic
 * search. Graph edges provide context: when you find a matching chunk,
 * you can traverse UP to its section/document for full context.
 *
 * Run: npx tsx examples/knowledge-base.ts
 */

import { PolyGraph, MemoryAdapter, cosineSimilarity, GraphQuery } from '../src/index'

const graph = new PolyGraph(new MemoryAdapter())
const now = Date.now()

// ────────────────────────────────────────────────────────────
// SEED: A mini knowledge base about databases
// ────────────────────────────────────────────────────────────

const docs: Record<string, { title: string; sections: Record<string, { chunks: { text: string; vec: number[] }[] }> }> = {
  'doc-1': {
    title: 'B-Tree Indexes',
    sections: {
      structure: {
        chunks: [
          { text: 'B-Trees are balanced search trees with high fanout', vec: [0.9, 0.1, 0.2, 0.1] },
          { text: 'Each node contains multiple keys and child pointers', vec: [0.8, 0.2, 0.3, 0.1] },
          { text: 'Leaf nodes contain pointers to actual data rows', vec: [0.7, 0.3, 0.2, 0.2] },
        ],
      },
      performance: {
        chunks: [
          { text: 'B-Tree height grows logarithmically with data size', vec: [0.6, 0.4, 0.5, 0.3] },
          { text: 'Typical B-Tree has fanout of several hundred', vec: [0.5, 0.5, 0.4, 0.2] },
        ],
      },
    },
  },
  'doc-2': {
    title: 'LSM Trees',
    sections: {
      design: {
        chunks: [
          { text: 'LSM trees use append-only writes with compaction', vec: [0.2, 0.9, 0.8, 0.1] },
          { text: 'Data is written to a memtable then flushed to SSTable', vec: [0.3, 0.8, 0.7, 0.2] },
          { text: 'Compaction merges SSTables and removes tombstones', vec: [0.2, 0.7, 0.6, 0.3] },
        ],
      },
      tradeoffs: {
        chunks: [
          { text: 'LSM trees have higher write throughput than B-Trees', vec: [0.4, 0.6, 0.9, 0.1] },
          { text: 'Read amplification is higher in LSM trees', vec: [0.3, 0.5, 0.8, 0.4] },
          { text: 'LSM trees are used by LevelDB, RocksDB, BigTable', vec: [0.5, 0.7, 0.5, 0.5] },
        ],
      },
    },
  },
  'doc-3': {
    title: 'Hash Indexes',
    sections: {
      basics: {
        chunks: [
          { text: 'Hash indexes map keys to slots via a hash function', vec: [0.8, 0.1, 0.1, 0.9] },
          { text: 'Collisions are resolved by chaining or open addressing', vec: [0.7, 0.2, 0.1, 0.8] },
        ],
      },
    },
  },
}

// Build the graph: document ──[CONTAINS, owned]──> section ──[CONTAINS, owned]──> chunk
let chunkCount = 0
for (const [docId, doc] of Object.entries(docs)) {
  graph.addNode({ id: docId, type: 'document', data: { title: doc.title }, insertedAt: now, updatedAt: now })

  for (const [secName, sec] of Object.entries(doc.sections)) {
    const secId = `${docId}:${secName}`
    graph.addNode({ id: secId, type: 'section', data: { heading: secName }, insertedAt: now, updatedAt: now })
    graph.addEdge(docId, 'CONTAINS', secId, {}, 'owned')

    for (let ci = 0; ci < sec.chunks.length; ci++) {
      const ch = sec.chunks[ci]
      const chId = `${secId}:chunk-${ci}`
      graph.addNode({
        id: chId, type: 'chunk', data: { text: ch.text },
        vector: new Float64Array(ch.vec), insertedAt: now, updatedAt: now,
      })
      graph.addEdge(secId, 'CONTAINS', chId, {}, 'owned')
      chunkCount++
    }
  }
}

console.log(`  Knowledge base: ${graph.size} nodes (${chunkCount} chunks across 3 documents)`)

// ────────────────────────────────────────────────────────────
// USE CASE 1: Semantic search — find relevant chunks
// ────────────────────────────────────────────────────────────

const queryVec = [0.3, 0.7, 0.8, 0.2] // "high write throughput storage engine"
const relevantChunks = graph.query()
  .whereNodeType('chunk')
  .similarTo(queryVec, 0.7)
  .toArray()

console.log(`\n  ── Semantic search: "high write throughput storage engine" ──`)
for (const ch of relevantChunks) {
  const score = cosineSimilarity(queryVec, ch.vector!)
  console.log(`    score=${score.toFixed(3)}  ${(ch.data as any).text.slice(0, 70)}...`)
}

// ────────────────────────────────────────────────────────────
// USE CASE 2: Context gathering via up-traversal
// ────────────────────────────────────────────────────────────
// For the top result, find its section and document context.

const topChunk = relevantChunks[0]
const sectionIds = graph.getEdgeSources(topChunk.id, 'CONTAINS')
const docIds = sectionIds.flatMap(sid => graph.getEdgeSources(sid, 'CONTAINS'))

console.log(`\n  ── Context for top result ──`)
console.log(`    Chunk:   ${(topChunk.data as any).text}`)
for (const secId of sectionIds) {
  const sec = graph.getNode(secId)
  console.log(`    Section: ${(sec?.data as any)?.heading}`)
}
for (const did of docIds) {
  const doc = graph.getNode(did)
  console.log(`    Document: ${(doc?.data as any)?.title}`)
}

// ────────────────────────────────────────────────────────────
// USE CASE 3: Hierarchical cascade — delete document
// ────────────────────────────────────────────────────────────
// Removing a document cascades to all sections and chunks.

console.log(`\n  ── Cascade: deleting doc-2 (LSM Trees) ──`)
graph.removeNode('doc-2')

// Check nothing survived
const lsmSurvivors = graph.query()
  .whereNodeType('chunk')
  .toArray()
  .filter(ch => ch.id.startsWith('doc-2'))

const lsmSections = graph.query()
  .whereNodeType('section')
  .toArray()
  .filter(s => s.id.startsWith('doc-2'))

console.log(`    Remaining doc-2 chunks:   ${lsmSurvivors.length}`)
console.log(`    Remaining doc-2 sections:  ${lsmSections.length}`)
console.log(`    Total nodes remaining:    ${graph.size} (was 23)`)

// ────────────────────────────────────────────────────────────
// USE CASE 4: Traverse down for full document
// ────────────────────────────────────────────────────────────

const doc1Content = graph.query()
  .where('title', 'B-Tree Indexes')
  .traverse('CONTAINS', 2, 'out') // doc → section → chunk
  .toArray()

console.log(`\n  ── Full content of B-Tree Indexes (BFS depth 2) ──`)
for (const node of doc1Content) {
  if (node.type === 'chunk') console.log(`    • ${(node.data as any).text}`)
  else if (node.type === 'section') console.log(`\n    [${(node.data as any).heading}]`)
}

console.log(`\n✓ Knowledge base examples completed`)

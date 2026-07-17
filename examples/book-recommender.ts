/**
 * Example: Book recommendation system
 *
 * Shows how polypack models a domain where graph edges (genre, author,
 * user-ratings) combine with vector embeddings (content similarity) and
 * ownership semantics (author owns their books).
 *
 * Run: npx tsx examples/book-recommender.ts
 */

import { PolyGraph, MemoryAdapter, cosineSimilarity, euclideanSimilarity } from '../src/index'
import type { DistanceFunction } from '../src/index'

// ────────────────────────────────────────────────────────────
// DOMAIN SETUP
// ────────────────────────────────────────────────────────────
// Embeddings: 5D [fiction_depth, romance, action, fantasy, literary]
// Computed from actual book content in a real system; here we hand-wave.

const graph = new PolyGraph(new MemoryAdapter())
const now = Date.now()

function addBook(id: string, title: string, author: string, genre: string, vec: number[]) {
  graph.addNode({
    id: `book:${id}`, type: 'book', data: { title, author, genre, year: 2024 },
    vector: new Float64Array(vec), insertedAt: now, updatedAt: now,
  })
  if (!graph.getNode(`author:${author}`)) {
    graph.addNode({
      id: `author:${author}`, type: 'author', data: { name: author },
      insertedAt: now, updatedAt: now,
    })
  }
  graph.addEdge(`author:${author}`, 'WROTE', `book:${id}`, {}, 'owned')
  graph.addEdge(`book:${id}`, 'IN_GENRE', `genre:${genre}`)
}

function addUser(id: string, name: string, prefVec: number[]) {
  graph.addNode({
    id: `user:${id}`, type: 'user', data: { name },
    vector: new Float64Array(prefVec), insertedAt: now, updatedAt: now,
  })
}

function addRating(userId: string, bookId: string, stars: number) {
  graph.addEdge(`user:${userId}`, 'RATED', `book:${bookId}`, { stars })
}

// ── Seed data ──

addBook('neuromancer', 'Neuromancer', 'Gibson', 'sci-fi', [0.90, 0.05, 0.80, 0.10, 0.85])
addBook('snow-crash', 'Snow Crash', 'Stephenson', 'sci-fi', [0.85, 0.10, 0.75, 0.15, 0.80])
addBook('dune', 'Dune', 'Herbert', 'sci-fi', [0.70, 0.30, 0.60, 0.50, 0.60])
addBook('lotr', 'The Fellowship of the Ring', 'Tolkien', 'fantasy', [0.40, 0.30, 0.70, 0.95, 0.90])
addBook('name-wind', 'The Name of the Wind', 'Rothfuss', 'fantasy', [0.50, 0.40, 0.50, 0.90, 0.85])
addBook('pride', 'Pride and Prejudice', 'Austen', 'romance', [0.30, 0.95, 0.10, 0.05, 0.95])
addBook('gatsby', 'The Great Gatsby', 'Fitzgerald', 'literary', [0.60, 0.70, 0.20, 0.10, 0.95])
addBook('three-body', 'The Three-Body Problem', 'Liu', 'sci-fi', [0.80, 0.05, 0.50, 0.60, 0.70])
addBook('hyperion', 'Hyperion', 'Simmons', 'sci-fi', [0.85, 0.15, 0.55, 0.65, 0.80])
addBook('stormlight', 'The Way of Kings', 'Sanderson', 'fantasy', [0.45, 0.20, 0.85, 0.92, 0.75])

addUser('alice', 'Alice (sci-fi fan)', [0.85, 0.10, 0.70, 0.20, 0.60])
addUser('bob', 'Bob (fantasy reader)', [0.40, 0.30, 0.60, 0.90, 0.80])
addUser('carol', 'Carol (literary taste)', [0.50, 0.80, 0.20, 0.15, 0.95])

addRating('alice', 'neuromancer', 5)
addRating('alice', 'snow-crash', 4)
addRating('alice', 'dune', 3)
addRating('bob', 'lotr', 5)
addRating('bob', 'name-wind', 4)
addRating('bob', 'stormlight', 5)
addRating('carol', 'pride', 5)
addRating('carol', 'gatsby', 4)
addRating('carol', 'dune', 2)

console.log(`\n  Graph: ${graph.size} nodes, ${graph.getEdges('author:Gibson').length + graph.getEdges('author:Tolkien').length + '...'} edges`)

// ────────────────────────────────────────────────────────────
// USE CASE 1: Content-based recommendations
// ────────────────────────────────────────────────────────────
// "Alice liked Neuromancer. What other books are similar?"

const neuro = graph.getNode('book:neuromancer')!
const neuroVec = [...neuro.vector!]

const similarBooks = graph.query()
  .whereNodeType('book')
  .similarTo(neuroVec, 0.75)
  .toArray()

console.log(`\n  ── USE CASE 1: Similar to Neuromancer (cosine ≥ 0.75) ──`)
for (const b of similarBooks) {
  const d = b.data as any
  const score = cosineSimilarity(neuroVec, b.vector!)
  console.log(`    ${d.title.padEnd(40)} score=${score.toFixed(3)}  [${d.genre}]`)
}

// ────────────────────────────────────────────────────────────
// USE CASE 2: Hybrid — similar books by genre + vector
// ────────────────────────────────────────────────────────────
// "Show me sci-fi books like Neuromancer, excluding ones I've rated"

const aliceRatings = graph.getEdgeTargets('user:alice', 'RATED')
console.log(`\n  ── USE CASE 2: Sci-fi similar to Neuromancer, unrated ──`)

const unratedSciFi = graph.query()
  .whereNodeType('book')
  .where('genre', 'sci-fi')
  .similarTo(neuroVec, 0.6)
  .toArray()
  .filter(b => !aliceRatings.includes(b.id))

for (const b of unratedSciFi) {
  const d = b.data as any
  console.log(`    ${d.title.padEnd(40)}  [unrated by Alice]`)
}

// ────────────────────────────────────────────────────────────
// USE CASE 3: Collaborative + content (recommend for user)
// ────────────────────────────────────────────────────────────
// "What books should Carol read?" — use her preference vector directly

const carol = graph.getNode('user:carol')!
const carolPref = [...carol.vector!]

const forCarol = graph.query()
  .whereNodeType('book')
  .where('genre', 'literary')
  .similarTo(carolPref, 0.5)
  .toArray()

console.log(`\n  ── USE CASE 3: Literary recommendations for Carol ──`)
for (const b of forCarol) {
  const d = b.data as any
  const score = cosineSimilarity(carolPref, b.vector!)
  console.log(`    ${d.title.padEnd(40)} score=${score.toFixed(3)}`)
}

// ────────────────────────────────────────────────────────────
// USE CASE 4: Author-centric discovery with cascade
// ────────────────────────────────────────────────────────────
// "Show me all books by Gibson, plus which ones Alice liked"
// Then: remove Gibson and show cascade worked.

const gibsonBooks = graph.getEdgeTargets('author:Gibson', 'WROTE')
console.log(`\n  ── USE CASE 4: Gibson's books ──`)
for (const bookId of gibsonBooks) {
  const b = graph.getNode(bookId)
  const d = b?.data as any
  const rated = aliceRatings.includes(bookId) ? '★ rated' : 'unrated'
  console.log(`    ${d?.title.padEnd(30)} ${rated}`)
}

console.log(`\n    Removing 'author:Gibson' (owned edges → cascade)...`)
graph.removeNode('author:Gibson')
for (const bookId of gibsonBooks) {
  const exists = graph.getNode(bookId)
  console.log(`    ${bookId}: ${exists ? 'SURVIVED' : 'cascade-deleted'}`)
}

// ────────────────────────────────────────────────────────────
// USE CASE 5: Cold-start recommendation (user preference vector)
// ────────────────────────────────────────────────────────────
// New user with no ratings — use their stated preference vector

console.log(`\n  ── USE CASE 5: Cold-start for new user (likes literary fiction) ──`)

const newUserPref = [0.40, 0.85, 0.10, 0.10, 0.90]
const coldStart = graph.query()
  .whereNodeType('book')
  .where('genre', 'literary')
  .similarTo(newUserPref)
  .toArray()

for (const b of coldStart) {
  const d = b.data as any
  const score = cosineSimilarity(newUserPref, b.vector!)
  console.log(`    ${d.title.padEnd(40)} score=${score.toFixed(3)}`)
}

console.log(`\n✓ Book recommender examples completed`)

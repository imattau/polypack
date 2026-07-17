import { describe, it, expect, beforeEach } from 'vitest'
import { PolyGraph } from '../src/graph'
import type { GroupedRow } from '../src/query'

describe('relational query extensions', () => {
  let graph: PolyGraph

  beforeEach(() => {
    graph = new PolyGraph()
    graph.addNode({ id: 'b1', type: 'book', data: { title: 'Neuromancer', genre: 'sci-fi', rating: 5, price: 12 }, insertedAt: 1, updatedAt: 1 })
    graph.addNode({ id: 'b2', type: 'book', data: { title: 'Dune', genre: 'sci-fi', rating: 4, price: 15 }, insertedAt: 2, updatedAt: 2 })
    graph.addNode({ id: 'b3', type: 'book', data: { title: 'LotR', genre: 'fantasy', rating: 5, price: 20 }, insertedAt: 3, updatedAt: 3 })
    graph.addNode({ id: 'b4', type: 'book', data: { title: 'Name of the Wind', genre: 'fantasy', rating: 4, price: 14 }, insertedAt: 4, updatedAt: 4 })
    graph.addNode({ id: 'b5', type: 'book', data: { title: 'Pride', genre: 'romance', rating: 5, price: 10 }, insertedAt: 5, updatedAt: 5 })
    graph.addNode({ id: 'u1', type: 'user', data: { name: 'Alice' }, insertedAt: 6, updatedAt: 6 })
    graph.addNode({ id: 'u2', type: 'user', data: { name: 'Bob' }, insertedAt: 7, updatedAt: 7 })
    graph.addEdge('u1', 'RATED', 'b1', { stars: 5 })
    graph.addEdge('u1', 'RATED', 'b2', { stars: 3 })
    graph.addEdge('u2', 'RATED', 'b3', { stars: 4 })
    graph.addEdge('u1', 'FOLLOWS', 'u2')
  })

  describe('pluck', () => {
    it('projects selected fields with id and type', () => {
      const rows = graph.query().whereNodeType('book').pluck('title', 'genre')
      expect(rows).toHaveLength(5)
      for (const row of rows) {
        expect(row).toHaveProperty('id')
        expect(row).toHaveProperty('type')
        expect(row).toHaveProperty('title')
        expect(row).toHaveProperty('genre')
        expect(row).not.toHaveProperty('rating')
      }
    })

    it('returns empty array when no matches', () => {
      const rows = graph.query().whereNodeType('nonexistent').pluck('title')
      expect(rows).toHaveLength(0)
    })
  })

  describe('aggregate', () => {
    it('sums a field', () => {
      const r = graph.query().whereNodeType('book').aggregate('price', 'sum')
      expect(r.value).toBe(71) // 12+15+20+14+10
      expect(r.count).toBe(5)
    })

    it('averages a field', () => {
      const r = graph.query().whereNodeType('book').aggregate('rating', 'avg')
      expect(r.value).toBeCloseTo(4.6)
      expect(r.count).toBe(5)
    })

    it('finds min and max', () => {
      const minR = graph.query().whereNodeType('book').aggregate('price', 'min')
      expect(minR.value).toBe(10)
      const maxR = graph.query().whereNodeType('book').aggregate('price', 'max')
      expect(maxR.value).toBe(20)
    })

    it('counts nodes with a non-null field', () => {
      const r = graph.query().whereNodeType('book').aggregate('price', 'count')
      expect(r.value).toBe(5)
    })

    it('honors where filters before aggregation', () => {
      const r = graph.query().whereNodeType('book').where('genre', 'sci-fi').aggregate('price', 'avg')
      expect(r.value).toBeCloseTo(13.5) // (12+15)/2
      expect(r.count).toBe(2)
    })

    it('returns zero for missing field', () => {
      // Add a book without price field
      graph.addNode({ id: 'b6', type: 'book', data: { title: 'Free Book' }, insertedAt: 8, updatedAt: 8 })
      const r = graph.query().whereNodeType('book').aggregate('price', 'sum')
      // Only counts nodes that have the field
      expect(r.value).toBe(71)
      expect(r.count).toBe(5)
    })
  })

  describe('groupAggregate', () => {
    it('groups by field and aggregates', () => {
      const rows = graph.query().whereNodeType('book').groupAggregate('price', 'avg', 'genre')
      expect(rows).toHaveLength(3) // sci-fi, fantasy, romance

      const sciFi = rows.find(r => r.key === 'sci-fi')!
      expect(sciFi.value).toBeCloseTo(13.5)
      expect(sciFi.count).toBe(2)

      const fantasy = rows.find(r => r.key === 'fantasy')!
      expect(fantasy.value).toBeCloseTo(17)
      expect(fantasy.count).toBe(2)

      const romance = rows.find(r => r.key === 'romance')!
      expect(romance.value).toBeCloseTo(10)
      expect(romance.count).toBe(1)
    })
  })

  describe('having', () => {
    it('filters groups after groupAggregate', () => {
      const rows = graph.query().whereNodeType('book').groupAggregate('price', 'avg', 'genre')
      const filtered = graph.query().having(rows, r => r.count >= 2)
      expect(filtered).toHaveLength(2) // sci-fi and fantasy have count >= 2
      expect(filtered.find(r => r.key === 'romance')).toBeUndefined()
    })
  })

  describe('join (chainable filter)', () => {
    it('filters users who have rated a book (outgoing edge)', () => {
      const users = graph.query()
        .whereNodeType('user')
        .join('RATED', 'out')
        .toArray()

      // Alice rated b1,b2; Bob rated b3 — both have at least one RATED edge
      expect(users).toHaveLength(2)
      const names = users.map(u => (u.data as any).name).sort()
      expect(names).toEqual(['Alice', 'Bob'])
    })

    it('filters users who rated a specific book via predicate', () => {
      const users = graph.query()
        .whereNodeType('user')
        .join('RATED', 'out', b => (b.data as any).title === 'Dune')
        .toArray()

      // Only Alice rated Dune
      expect(users).toHaveLength(1)
      expect(users[0].data.name).toBe('Alice')
    })

    it('filters books that have been rated (incoming edge)', () => {
      const books = graph.query()
        .whereNodeType('book')
        .join('RATED', 'in')
        .toArray()

      // b1,b2,b3 have incoming RATED edges; b4,b5 do not
      expect(books).toHaveLength(3)
      const titles = books.map(b => (b.data as any).title).sort()
      expect(titles).toEqual(['Dune', 'LotR', 'Neuromancer'])
    })

    it('filters books rated by a specific user via predicate', () => {
      const books = graph.query()
        .whereNodeType('book')
        .join('RATED', 'in', u => (u.data as any).name === 'Alice')
        .toArray()

      // Alice rated Neuromancer and Dune
      expect(books).toHaveLength(2)
      const titles = books.map(b => (b.data as any).title).sort()
      expect(titles).toEqual(['Dune', 'Neuromancer'])
    })

    it('returns empty when no edges match', () => {
      const result = graph.query()
        .whereNodeType('user')
        .join('NONEXISTENT', 'out')
        .toArray()

      expect(result).toHaveLength(0)
    })

    it('composes with other filters and aggregations', () => {
      // Average price of books rated by users (all books with incoming RATED)
      const r = graph.query()
        .whereNodeType('book')
        .join('RATED', 'in')
        .aggregate('price', 'avg')

      // b1(12) + b2(15) + b3(20) / 3
      expect(r.value).toBeCloseTo(15.667, 2)
      expect(r.count).toBe(3)
    })
  })

  describe('groupByVector (vector clustering)', () => {
    beforeEach(() => {
      // Add vectors to the existing books (they don't have them yet)
      // Books: b1=Neuromancer, b2=Dune, b3=LotR, b4=Name of the Wind, b5=Pride
      const b1 = graph.getNode('b1')!; b1.vector = new Float64Array([0.9, 0.2, 0.1])
      const b2 = graph.getNode('b2')!; b2.vector = new Float64Array([0.7, 0.4, 0.2])
      const b3 = graph.getNode('b3')!; b3.vector = new Float64Array([0.3, 0.8, 0.7])
      const b4 = graph.getNode('b4')!; b4.vector = new Float64Array([0.2, 0.7, 0.9])
      const b5 = graph.getNode('b5')!; b5.vector = new Float64Array([0.9, 0.6, 0.3])
    })

    it('groups by nearest centroid and aggregates', () => {
      const groups = graph.query()
        .whereNodeType('book')
        .groupByVector(
          [
            { key: 'tech', centroid: [0.9, 0.2, 0.1] },
            { key: 'fantasy', centroid: [0.2, 0.8, 0.8] },
          ],
          'price', 'avg', 0.3,
        )

      expect(groups.length).toBeGreaterThanOrEqual(2)
      const tech = groups.find(g => g.key === 'tech')!
      expect(tech.count).toBeGreaterThanOrEqual(2) // Neuromancer(12) + Dune(15) + Pride(10) = 3 or 2
    })

    it('uncategorizes nodes below threshold', () => {
      // Very high threshold — only near-identical vectors match
      const groups = graph.query()
        .whereNodeType('book')
        .groupByVector(
          [{ key: 'exact', centroid: [0.9, 0.2, 0.1] }],
          'title', 'count', 0.99,
        )

      const exact = groups.find(g => g.key === 'exact')!
      expect(exact.count).toBe(1) // Only b1 (Neuromancer) matches exactly-ish
    })

    it('handles nodes without vectors gracefully', () => {
      // Add a book without a vector
      graph.addNode({ id: 'b6', type: 'book', data: { title: 'NoVec', price: 5 }, insertedAt: 9, updatedAt: 9 })

      const groups = graph.query()
        .whereNodeType('book')
        .groupByVector(
          [{ key: 'all', centroid: [0.5, 0.5, 0.5] }],
          'price', 'sum', 0,
        )

      // b6 has no vector so it's excluded from vector grouping
      const all = groups.find(g => g.key === 'all')!
      expect(all.count).toBe(5) // only the 5 books with vectors
    })
  })

  describe('GraphQuery gap tests', () => {
    it('whereAttributeRange filters by numeric range', () => {
      const results = graph.query()
        .whereNodeType('book')
        .whereAttributeRange('price', { above: 13, below: 20 })
        .toArray()
      // Books with price > 13 and < 20: Name of the Wind(14), Dune(15)
      expect(results).toHaveLength(2)
    })

    it('whereEdgeSource filters nodes targeted by a source', () => {
      // Books that are the target of a RATED edge from user u1 (Alice)
      // whereEdgeSource('u1') checks if u1 has an edge targeting the node
      // (no additional whereEdge needed — books don't have outgoing RATED)
      const books = graph.query()
        .whereNodeType('book')
        .whereEdgeSource('u1')
        .toArray()
      expect(books).toHaveLength(2) // Alice rated b1, b2
    })

    it('offset skips results', () => {
      const results = graph.query()
        .whereNodeType('book')
        .orderBy('price', 'asc')
        .offset(2)
        .pluck('title', 'price')
      expect(results).toHaveLength(3) // 5 total - 2 offset = 3
      // Sorted asc: 10(Pride), 12(Neuromancer), 14(Name), 15(Dune), 20(LotR)
      // Offset 2: 14(Name of the Wind), 15(Dune), 20(LotR)
      expect(results[0].title).toBe('Name of the Wind')
    })

    it('offset + limit combined pagination', () => {
      const results = graph.query()
        .whereNodeType('book')
        .orderBy('price', 'asc')
        .offset(1)
        .limit(2)
        .pluck('title')
      expect(results).toHaveLength(2)
    })

    it('count returns matching count', () => {
      const c = graph.query().whereNodeType('book').where('genre', 'sci-fi').count()
      expect(c).toBe(2)
    })

    it('count with traversal includes seeds + traversed', () => {
      // 2 users (seeds) + 3 books (traversed) = 5
      const c = graph.query().whereNodeType('user').traverse('RATED', 1, 'out').count()
      expect(c).toBe(5)
    })

    it('count with no matches returns 0', () => {
      const c = graph.query().whereNodeType('nonexistent').count()
      expect(c).toBe(0)
    })

    it('first returns first match or null', () => {
      const first = graph.query().whereNodeType('book').orderBy('price', 'asc').first()
      expect(first).not.toBeNull()
      expect((first!.data as any).price).toBe(10) // Pride

      const none = graph.query().whereNodeType('nonexistent').first()
      expect(none).toBeNull()
    })

    it('collect returns connected nodes via edges', () => {
      // Users connected to book b1 via RATED (outgoing from user)
      const usersWhoRated = graph.query()
        .whereNodeType('user')
        .collect('RATED', 'out')
      expect(usersWhoRated.length).toBeGreaterThan(0)
    })

    it('collect with in direction returns source nodes', () => {
      // collect with 'in' returns nodes on the source side of edges
      // For book nodes, 'in' on RATED = the users who rated them
      const users = graph.query()
        .whereNodeType('book')
        .collect('RATED', 'in')
      expect(users).toHaveLength(2) // Alice and Bob
    })

    it('traverse with in direction includes seed + sources', () => {
      // traverse with 'in' on Dune: seed(Dune) + source(Alice) = 2
      const users = graph.query()
        .whereNodeType('book')
        .where('title', 'Dune')
        .traverse('RATED', 1, 'in')
        .toArray()
      expect(users).toHaveLength(2) // Dune + Alice
      const names = users.map(u => (u.data as any).name).filter(Boolean)
      expect(names).toEqual(['Alice'])
    })

    it('orderBy desc sorts descending', () => {
      const results = graph.query()
        .whereNodeType('book')
        .orderBy('price', 'desc')
        .pluck('title', 'price')
      expect(results[0].title).toBe('LotR')    // 20
      expect(results[1].title).toBe('Dune')    // 15
      expect(results[4].title).toBe('Pride')   // 10
    })

    it('similarTo + orderBy composes (similarTo wins)', () => {
      graph.addNode({ id: 'x1', type: 't', data: { sortKey: 1 }, vector: new Float64Array([1, 0]), insertedAt: 1, updatedAt: 1 })
      graph.addNode({ id: 'x2', type: 't', data: { sortKey: 2 }, vector: new Float64Array([0.9, 0.1]), insertedAt: 2, updatedAt: 2 })

      const results = graph.query()
        .whereNodeType('t')
        .orderBy('sortKey', 'desc')
        .similarTo([1, 0], 0.5)
        .toArray()
      // similarTo runs after orderBy, so the final order is by similarity, not sortKey
      expect(results[0].id).toBe('x1')
      expect(results[1].id).toBe('x2')
    })

    it('ids returns only IDs', () => {
      const ids = graph.query().whereNodeType('book').where('genre', 'sci-fi').ids()
      expect(ids).toEqual(['b1', 'b2'])
    })

    it('uniqueKeys returns distinct field values across all nodes', () => {
      const genres = graph.query().uniqueKeys('genre') as string[]
      expect(genres.sort()).toEqual(['fantasy', 'romance', 'sci-fi'])
    })

    it('whereAttribute (alias for where) works', () => {
      const results = graph.query().whereNodeType('book').whereAttribute('title', 'Dune').toArray()
      expect(results).toHaveLength(1)
    })
  })
})

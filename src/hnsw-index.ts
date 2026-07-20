import { cosineSimilarity } from './vector-index.js'
import type { DistanceFunction } from './vector-index.js'
import { assertFiniteVector, assertNonNegativeInteger } from './utils.js'

export interface HNSWConfig {
  M?: number
  Mmax0?: number
  efConstruction?: number
  efSearch?: number
}

interface Candidate {
  id: string
  dist: number
}

const DEF: Required<HNSWConfig> = {
  M: 16,
  Mmax0: 32,
  efConstruction: 200,
  efSearch: 200,
}

export class HNSWIndex {
  private nodes = new Map<string, number[]>()
  private nodeLevel = new Map<string, number>()
  private adjacency = new Map<number, Map<string, Set<string>>>()
  private entryPoint: string | null = null
  private maxLayer = -1
  private removed = new Set<string>()
  private onChange?: (id: string) => void
  private distanceFn: DistanceFunction
  private config: Required<HNSWConfig>
  private mL: number

  constructor(
    onChange?: (id: string) => void,
    distanceFn?: DistanceFunction,
    config?: HNSWConfig,
  ) {
    this.onChange = onChange
    this.distanceFn = distanceFn ?? cosineSimilarity
    this.config = { ...DEF, ...config }
    this.mL = 1 / Math.log(Math.max(this.config.M, 2))
  }

  // ── Public API ──

  add(id: string, vector: number[]): void {
    if (!id) throw new TypeError('Vector id must not be empty')
    assertFiniteVector(vector)
    const level = this.assignLevel()
    this.nodes.set(id, [...vector])
    this.insertIntoGraph(id, level)
    this.onChange?.(id)
  }

  hydrate(id: string, vector: number[]): void {
    if (!id) throw new TypeError('Vector id must not be empty')
    assertFiniteVector(vector)
    const level = this.assignLevel()
    this.nodes.set(id, [...vector])
    this.insertIntoGraph(id, level)
  }

  addMany(entries: Array<{ id: string; vector: number[] }>): void {
    for (const { id, vector } of entries) {
      if (!id) throw new TypeError('Vector id must not be empty')
      assertFiniteVector(vector)
      const level = this.assignLevel()
      this.nodes.set(id, [...vector])
      this.insertIntoGraph(id, level)
      this.onChange?.(id)
    }
  }

  remove(id: string): void {
    if (!this.nodes.has(id)) return
    this.removed.add(id)
    this.nodes.delete(id)
    this.nodeLevel.delete(id)
    if (this.entryPoint === id) {
      this.entryPoint = this.pickNewEntryPoint()
    }
  }

  removeMany(ids: string[]): void {
    for (const id of ids) this.remove(id)
  }

  query(
    vector: number[],
    topK: number,
    threshold = 0,
  ): Array<{ id: string; score: number }> {
    assertFiniteVector(vector, 'query vector')
    assertNonNegativeInteger(topK, 'topK')
    if (!Number.isFinite(threshold)) throw new RangeError('threshold must be finite')
    if (topK === 0 || this.nodes.size === 0) return []

    const ep = this.entryPoint
    if (!ep) return []

    let current = ep
    for (let lc = this.maxLayer; lc >= 1; lc--) {
      current = this.greedySearch(vector, current, lc)
    }

    const candidates = this.searchLayer(vector, current, this.config.efSearch, 0)

    const scored = candidates
      .map(({ id, dist }) => ({ id, score: 1 - dist }))
      .filter(r => r.score >= threshold)

    return scored.slice(0, topK).map((r, i) => ({ id: r.id, score: r.score }))
  }

  clear(): void {
    this.nodes.clear()
    this.nodeLevel.clear()
    this.adjacency.clear()
    this.removed.clear()
    this.entryPoint = null
    this.maxLayer = -1
  }

  get size(): number {
    return this.nodes.size
  }

  *entries(): IterableIterator<[string, number[]]> {
    for (const [id, vector] of this.nodes) {
      if (!this.removed.has(id)) yield [id, [...vector]]
    }
  }

  has(id: string): boolean {
    return this.nodes.has(id)
  }

  get(id: string): number[] | undefined {
    const vector = this.nodes.get(id)
    return vector ? [...vector] : undefined
  }

  // ── Internal helpers ──

  private dist(a: number[], b: number[]): number {
    if (a.length !== b.length) {
      throw new RangeError(`Vector dimension mismatch: ${a.length} !== ${b.length}`)
    }
    return 1 - this.distanceFn(a, b)
  }

  private assignLevel(): number {
    return Math.floor(-Math.log(Math.random()) * this.mL)
  }

  private ensureLayer(layer: number): void {
    if (!this.adjacency.has(layer)) {
      this.adjacency.set(layer, new Map())
    }
  }

  private getNeighbors(nodeId: string, layer: number): string[] {
    const layerAdj = this.adjacency.get(layer)
    if (!layerAdj) return []
    const neighbors = layerAdj.get(nodeId)
    if (!neighbors) return []
    return [...neighbors].filter(nid => !this.removed.has(nid) && this.nodes.has(nid))
  }

  private pickNewEntryPoint(): string | null {
    for (let lc = this.maxLayer; lc >= 0; lc--) {
      const layerAdj = this.adjacency.get(lc)
      if (!layerAdj) continue
      for (const nid of layerAdj.keys()) {
        if (!this.removed.has(nid) && this.nodes.has(nid)) return nid
      }
    }
    return null
  }

  // ── HNSW algorithms ──

  private greedySearch(q: number[], ep: string, layer: number): string {
    let current = ep
    let currentDist = this.dist(q, this.nodes.get(current)!)
    const visited = new Set<string>([current])

    while (true) {
      const neighbors = this.getNeighbors(current, layer)
      let improved = false
      for (const nid of neighbors) {
        if (visited.has(nid)) continue
        visited.add(nid)
        const nDist = this.dist(q, this.nodes.get(nid)!)
        if (nDist < currentDist) {
          current = nid
          currentDist = nDist
          improved = true
        }
      }
      if (!improved) break
    }

    return current
  }

  private searchLayer(
    q: number[],
    entry: string,
    ef: number,
    layer: number,
  ): Candidate[] {
    const visited = new Set<string>([entry])
    const entryDist = this.dist(q, this.nodes.get(entry)!)

    const candidates: Candidate[] = [{ id: entry, dist: entryDist }]
    const results: Candidate[] = [{ id: entry, dist: entryDist }]

    const minCmp = (a: Candidate, b: Candidate) => a.dist < b.dist
    const maxCmp = (a: Candidate, b: Candidate) => a.dist > b.dist

    while (candidates.length > 0) {
      const c = popHeap(candidates, minCmp)!
      const f = results[0]

      if (c.dist > f.dist) break

      const neighbors = this.getNeighbors(c.id, layer)
      for (const eid of neighbors) {
        if (visited.has(eid)) continue
        visited.add(eid)

        const eDist = this.dist(q, this.nodes.get(eid)!)

        if (results.length < ef || eDist < results[0].dist) {
          pushHeap(candidates, minCmp, { id: eid, dist: eDist })
          pushHeap(results, maxCmp, { id: eid, dist: eDist })
          if (results.length > ef) {
            popHeap(results, maxCmp)
          }
        }
      }
    }

    results.sort((a, b) => a.dist - b.dist)
    if (results.length > ef) results.length = ef
    return results
  }

  private insertIntoGraph(id: string, level: number): void {
    if (this.nodes.size === 1) {
      this.entryPoint = id
      this.maxLayer = level
      this.nodeLevel.set(id, level)
      this.ensureLayer(level)
      this.adjacency.get(level)!.set(id, new Set())
      return
    }

    let ep = this.entryPoint!

    for (let lc = this.maxLayer; lc > level; lc--) {
      ep = this.greedySearch(this.nodes.get(id)!, ep, lc)
    }

    for (let lc = Math.min(level, this.maxLayer); lc >= 0; lc--) {
      const candidates = this.searchLayer(
        this.nodes.get(id)!,
        ep,
        this.config.efConstruction,
        lc,
      )

      const maxConn = lc === 0 ? this.config.Mmax0 : this.config.M
      const neighborIds = this.selectNeighborsSimple(candidates, maxConn)

      this.ensureLayer(lc)
      const layerAdj = this.adjacency.get(lc)!

      if (!layerAdj.has(id)) layerAdj.set(id, new Set())
      const mySet = layerAdj.get(id)!

      for (const nid of neighborIds) {
        mySet.add(nid)
        if (!layerAdj.has(nid)) layerAdj.set(nid, new Set())
        layerAdj.get(nid)!.add(id)
      }

      for (const nid of neighborIds) {
        this.shrinkNeighbors(nid, lc, maxConn)
      }

      ep = candidates[0].id
    }

    if (level > this.maxLayer) {
      this.maxLayer = level
      this.entryPoint = id
    }

    this.nodeLevel.set(id, level)
  }

  private selectNeighborsSimple(candidates: Candidate[], M: number): string[] {
    return candidates.slice(0, M).map(c => c.id)
  }

  private shrinkNeighbors(nid: string, layer: number, max: number): void {
    const layerAdj = this.adjacency.get(layer)
    if (!layerAdj) return
    const neighbors = layerAdj.get(nid)
    if (!neighbors || neighbors.size <= max) return

    const vec = this.nodes.get(nid)
    if (!vec) return

    const withDist: Candidate[] = []
    for (const neighbor of neighbors) {
      const nVec = this.nodes.get(neighbor)
      if (nVec) withDist.push({ id: neighbor, dist: this.dist(vec, nVec) })
    }

    withDist.sort((a, b) => a.dist - b.dist)
    neighbors.clear()
    for (let i = 0; i < Math.min(withDist.length, max); i++) {
      neighbors.add(withDist[i].id)
    }
  }
}

function pushHeap<T>(heap: T[], cmp: (a: T, b: T) => boolean, item: T): void {
  heap.push(item)
  let i = heap.length - 1
  while (i > 0) {
    const p = (i - 1) >> 1
    if (!cmp(heap[i], heap[p])) break
    ;[heap[i], heap[p]] = [heap[p], heap[i]]
    i = p
  }
}

function popHeap<T>(heap: T[], cmp: (a: T, b: T) => boolean): T | undefined {
  if (heap.length === 0) return undefined
  const result = heap[0]
  const last = heap.pop()!
  if (heap.length > 0) {
    heap[0] = last
    let i = 0
    while (true) {
      let smallest = i
      const l = i * 2 + 1
      const r = l + 1
      if (l < heap.length && cmp(heap[l], heap[smallest])) smallest = l
      if (r < heap.length && cmp(heap[r], heap[smallest])) smallest = r
      if (smallest === i) break
      ;[heap[i], heap[smallest]] = [heap[smallest], heap[i]]
      i = smallest
    }
  }
  return result
}

export type DistanceFunction = (a: ArrayLike<number>, b: ArrayLike<number>) => number

function minLen(a: ArrayLike<number>, b: ArrayLike<number>): number {
  return Math.min(a.length, b.length)
}

export function cosineSimilarity(a: ArrayLike<number>, b: ArrayLike<number>): number {
  const len = minLen(a, b)
  let dot = 0
  let na = 0
  let nb = 0
  for (let i = 0; i < len; i++) {
    dot += a[i] * b[i]
    na += a[i] * a[i]
    nb += b[i] * b[i]
  }
  const denom = Math.sqrt(na) * Math.sqrt(nb)
  return denom === 0 ? 0 : dot / denom
}

export function euclideanSimilarity(a: ArrayLike<number>, b: ArrayLike<number>): number {
  const len = minLen(a, b)
  let sum = 0
  for (let i = 0; i < len; i++) {
    const diff = a[i] - b[i]
    sum += diff * diff
  }
  return 1 / (1 + Math.sqrt(sum))
}

export class VectorIndex {
  private vectors = new Map<string, number[]>()
  private onChange?: (id: string) => void
  private distanceFn: DistanceFunction

  constructor(onChange?: (id: string) => void, distanceFn?: DistanceFunction) {
    this.onChange = onChange
    this.distanceFn = distanceFn ?? cosineSimilarity
  }

  add(id: string, vector: number[]): void {
    this.vectors.set(id, vector)
    this.onChange?.(id)
  }

  addMany(entries: Array<{ id: string; vector: number[] }>): void {
    for (const { id, vector } of entries) {
      this.vectors.set(id, vector)
      this.onChange?.(id)
    }
  }

  remove(id: string): void {
    this.vectors.delete(id)
  }

  removeMany(ids: string[]): void {
    for (const id of ids) {
      this.vectors.delete(id)
    }
  }

  query(
    vector: number[],
    topK: number,
    threshold = 0
  ): Array<{ id: string; score: number }> {
    const results: Array<{ id: string; score: number }> = []
    for (const [id, v] of this.vectors) {
      const score = this.distanceFn(vector, v)
      if (score < threshold) continue
      results.push({ id, score })
    }
    results.sort((a, b) => b.score - a.score)
    return results.slice(0, topK)
  }

  clear(): void {
    this.vectors.clear()
  }

  get size(): number {
    return this.vectors.size
  }

  entries(): IterableIterator<[string, number[]]> {
    return this.vectors.entries()
  }

  has(id: string): boolean {
    return this.vectors.has(id)
  }

  get(id: string): number[] | undefined {
    return this.vectors.get(id)
  }
}

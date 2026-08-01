import { Subscription } from 'rxjs'
import type { PolyGraph } from './graph.js'
import type { PolyNode, NodeActivation } from './types.js'
import { ACTIVATION_DEFAULTS, clamp01, decayActivationState, decayFactor } from './utils.js'

const HOUR = 3_600_000
const DAY = 24 * HOUR

const DEFAULT_CONFIG = {
  scoreHalfLifeMs: ACTIVATION_DEFAULTS.scoreHalfLifeMs,
  importanceHalfLifeMs: ACTIVATION_DEFAULTS.importanceHalfLifeMs,
  importanceGain: ACTIVATION_DEFAULTS.importanceGain,
  spreadDecay: 0.5,
  spreadDepth: 2,
  recencyHalfLifeMs: 7 * DAY,
  weights: { semantic: 1, graph: 1, recency: 1, usage: 1 },
  minReinforceDelta: 0.05,
  pulseThreshold: 0,
  absorbThreshold: 0.3,
  absorbGain: 0.05,
} as const

export interface ActivationConfig {
  /** Half-life of the short-term `score` curve. Default 24h. */
  scoreHalfLifeMs?: number
  /** Half-life of the long-term `importance` curve. Default 30 days. */
  importanceHalfLifeMs?: number
  /** How much of a reinforcement delta is folded into `importance` (0–1). Default 0.05. */
  importanceGain?: number
  /** Per-hop attenuation for spreading activation. Default 0.5. */
  spreadDecay?: number
  /** Maximum hop distance for spreading activation. Default 2. */
  spreadDepth?: number
  /** Half-life used to fold a node's age into a composite pulse score. Default 7 days. */
  recencyHalfLifeMs?: number
  /** Weights for the composite pulse score. Defaults to all 1. */
  weights?: { semantic?: number; graph?: number; recency?: number; usage?: number }
  /** Attention changes below this magnitude never become durable reinforcement. Default 0.05. */
  minReinforceDelta?: number
  /** Minimum composite score kept by `pulse`. Default 0. */
  pulseThreshold?: number
  /** Composite score at which `absorb` reinforces a node. Default 0.3. */
  absorbThreshold?: number
  /** Reinforcement delta applied by `absorb` (`gain * composite`). Default 0.05. */
  absorbGain?: number
}

export interface SpreadOptions {
  depth?: number
  decay?: number
  /** Restrict traversal to these edge types. Default: all edge types. */
  edgeTypes?: string[]
}

export interface PulseOptions extends SpreadOptions {
  threshold?: number
  /** Cap on the number of semantic seeds. Default: all loaded vectors. */
  topK?: number
  /** Minimum composite score kept by `pulse`. Defaults to `config.pulseThreshold`. */
  pulseThreshold?: number
  /**
   * Minimum semantic similarity for a node to seed the activated region.
   * Default 0 — nodes with zero similarity to the query never enter the region.
   */
  semanticThreshold?: number
}

export type VectorLike = number[] | Float32Array | Float64Array

/** Rank a score map descending, discarding entries at or below `threshold`. */
function ranked(scores: Map<string, number>, threshold: number): Map<string, number> {
  return new Map(
    [...scores.entries()]
      .filter(([, score]) => score > threshold)
      .sort((a, b) => b[1] - a[1]),
  )
}

/**
 * Merge two durable activation records (e.g. when a full node payload arrives
 * from sync). Decay-corrects both to `now`, keeps the stronger component of
 * each, and re-anchors the result to `now` so future decay is self-consistent.
 *
 * This is the **total-state** merge (max, idempotent for re-delivered
 * snapshots). Concurrent *deltas* are handled additively by the `activationUpdate`
 * sync op instead — `0.2 + 0.3 = 0.5`, not last-write-wins.
 */
export function mergeActivation(
  existing: NodeActivation,
  incoming: NodeActivation,
  now = Date.now(),
): NodeActivation {
  const ex = decayActivationState(existing, now)
  const inc = decayActivationState(incoming, now)
  return {
    score: Math.max(ex.score, inc.score),
    importance: Math.max(ex.importance, inc.importance),
    reinforcementCount: Math.max(existing.reinforcementCount, incoming.reinforcementCount),
    lastMeaningfulActivation: now,
  }
}

/**
 * Composes the adaptive "activation" layer over a {@link PolyGraph}.
 *
 * Two tiers:
 * - **Durable** — `reinforce`/`reinforceAll` write `NodeActivation` through
 *   {@link PolyGraph.reinforceNode}, so it persists and replicates (additively,
 *   via the `activationUpdate` sync op).
 * - **Transient** — `bumpAttention`/`attentionOf` hold runtime-only attention in
 *   this engine, never serialized or synced. `effective` merges the two.
 *
 * `spread` implements graph-based spreading activation, `pulse`/`absorb`
 * implement the semantic + relational + recency + usage composite, and
 * `workingMemory` materializes the current "mental state" (most active nodes).
 */
export class ActivationEngine {
  readonly graph: PolyGraph
  readonly config: Required<Pick<ActivationConfig, 'scoreHalfLifeMs' | 'importanceHalfLifeMs' | 'importanceGain' | 'spreadDecay' | 'spreadDepth' | 'recencyHalfLifeMs' | 'minReinforceDelta' | 'pulseThreshold' | 'absorbThreshold' | 'absorbGain'>>
  readonly weights: Required<NonNullable<ActivationConfig['weights']>>

  private attention = new Map<string, number>()
  private subscription: Subscription

  constructor(graph: PolyGraph, config: ActivationConfig = {}) {
    this.graph = graph
    this.config = {
      scoreHalfLifeMs: config.scoreHalfLifeMs ?? DEFAULT_CONFIG.scoreHalfLifeMs,
      importanceHalfLifeMs: config.importanceHalfLifeMs ?? DEFAULT_CONFIG.importanceHalfLifeMs,
      importanceGain: config.importanceGain ?? DEFAULT_CONFIG.importanceGain,
      spreadDecay: config.spreadDecay ?? DEFAULT_CONFIG.spreadDecay,
      spreadDepth: config.spreadDepth ?? DEFAULT_CONFIG.spreadDepth,
      recencyHalfLifeMs: config.recencyHalfLifeMs ?? DEFAULT_CONFIG.recencyHalfLifeMs,
      minReinforceDelta: config.minReinforceDelta ?? DEFAULT_CONFIG.minReinforceDelta,
      pulseThreshold: config.pulseThreshold ?? DEFAULT_CONFIG.pulseThreshold,
      absorbThreshold: config.absorbThreshold ?? DEFAULT_CONFIG.absorbThreshold,
      absorbGain: config.absorbGain ?? DEFAULT_CONFIG.absorbGain,
    }
    this.weights = { ...DEFAULT_CONFIG.weights, ...config.weights }
    this.subscription = graph.changes.subscribe((event) => {
      if (event.type === 'node_removed' && event.nodeId) this.attention.delete(event.nodeId)
    })
  }

  /** Stop observing the graph and drop transient attention. Durable state is untouched. */
  dispose(): void {
    this.subscription.unsubscribe()
    this.attention.clear()
  }

  // ── Transient attention (local, never synced) ──

  /**
   * Add a runtime-only attention delta. Attention accumulates locally and is
   * promoted to durable reinforcement once it clears `minReinforceDelta`, so
   * tiny events (scrolls, focus) stay local while meaningful ones become
   * persisted and synced.
   */
  bumpAttention(id: string, amount: number): void {
    const next = clamp01((this.attention.get(id) ?? 0) + amount)
    this.attention.set(id, next)
    if (next >= this.config.minReinforceDelta) {
      const promoted = next
      this.attention.delete(id)
      this.graph.reinforceNode(id, promoted, 'attention')
    }
  }

  /** Current transient attention for `id` (0 when none). */
  attentionOf(id: string): number {
    return this.attention.get(id) ?? 0
  }

  /** Durable decayed score plus any transient attention. */
  effective(id: string): number {
    const state = this.graph.getActivationState(id)
    if (!state) return this.attentionOf(id)
    const decayed = decayActivationState(state, Date.now(), this.config.scoreHalfLifeMs, this.config.importanceHalfLifeMs)
    return clamp01(decayed.score + this.attentionOf(id))
  }

  // ── Durable reinforcement (persisted and synced additively) ──

  /** Reinforce a loaded node's durable activation. See {@link PolyGraph.reinforceNode}. */
  reinforce(id: string, amount: number, reason?: string): void {
    this.graph.reinforceNode(id, amount, reason)
  }

  /** Reinforce several nodes, coalescing change events into one flush. */
  reinforceAll(entries: Array<{ id: string; amount: number; reason?: string }>): void {
    if (entries.length === 0) return
    const graph = this.graph
    graph.startBatch()
    try {
      for (const entry of entries) graph.reinforceNode(entry.id, entry.amount, entry.reason)
    } finally {
      graph.endBatch()
    }
  }

  // ── Relational spreading activation ──

  /**
   * Spread activation outward from `seeds` across outgoing edges. Each hop
   * attenuates the contribution by `decay` (default `config.spreadDecay`), and
   * multiple paths to a node sum. Returns `{ nodeId: contribution }`.
   */
  spread(seeds: string[], options: SpreadOptions = {}): Map<string, number> {
    const depth = options.depth ?? this.config.spreadDepth
    const decay = options.decay ?? this.config.spreadDecay
    const edgeTypes = options.edgeTypes
    const contributions = new Map<string, number>()
    if (depth <= 0 || seeds.length === 0) return contributions
    const visited = new Set(seeds)
    let frontier = [...seeds]
    for (let hop = 0; hop < depth; hop++) {
      const next: string[] = []
      for (const id of frontier) {
        const edges = this.graph.getEdges(id).filter(e => !edgeTypes || edgeTypes.includes(e.type))
        for (const edge of edges) {
          if (visited.has(edge.target)) continue
          visited.add(edge.target)
          contributions.set(edge.target, (contributions.get(edge.target) ?? 0) + Math.pow(decay, hop + 1))
          next.push(edge.target)
        }
      }
      frontier = next
    }
    return contributions
  }

  // ── Composite scoring ──

  /** Decay-normalised usage signal: a node's long-term `importance` (0 when unreinforced). */
  usageOf(node: PolyNode): number {
    return node.activation ? node.activation.importance : 0
  }

  /** Composite score for a node given a semantic hit and a graph contribution. */
  scoreOf(node: PolyNode, semantic: number, graphContribution: number, now = Date.now()): number {
    const recency = decayFactor(now - node.insertedAt, this.config.recencyHalfLifeMs)
    return (
      this.weights.semantic * semantic +
      this.weights.graph * graphContribution +
      this.weights.recency * recency +
      this.weights.usage * this.usageOf(node)
    )
  }

  // ── Semantic pulse / absorb ──

  private async toVector(input: string | VectorLike): Promise<number[]> {
    if (typeof input === 'string') return [...(await this.graph.embed(input))]
    return [...input]
  }

  /**
   * Score the region of the graph around `input` (text or vector): semantic
   * seeds via vector similarity, outward spreading activation, folded with
   * recency and usage. Read-only — returns `{ nodeId: composite }` ranked
   * descending for loaded nodes.
   */
  async pulse(input: string | VectorLike, options: PulseOptions = {}): Promise<Map<string, number>> {
    const graph = this.graph
    const vector = await this.toVector(input)
    const threshold = options.pulseThreshold ?? this.config.pulseThreshold
    const semanticThreshold = Math.max(options.semanticThreshold ?? 0, 0)
    const now = Date.now()

    const semantic = new Map<string, number>()
    const topK = options.topK ?? graph.vectors.size
    for (const hit of graph.vectors.query(vector, topK, 0)) {
      // Zero (or negative) similarity never seeds the activated region.
      if (hit.score > semanticThreshold) semantic.set(hit.id, hit.score)
    }

    const seeds = [...semantic.keys()]
    const graphContributions = this.spread(seeds, options)

    const scores = new Map<string, number>()
    const ids = new Set([...semantic.keys(), ...graphContributions.keys()])
    for (const id of ids) {
      const node = graph.getNode(id)
      if (!node) continue
      scores.set(id, this.scoreOf(node, semantic.get(id) ?? 0, graphContributions.get(id) ?? 0, now))
    }
    return ranked(scores, threshold)
  }

  /**
   * `pulse` plus reinforcement: nodes whose composite score clears
   * `absorbThreshold` receive durable reinforcement of `absorbGain * score`.
   * This is the self-maintaining "engine" — reading/querying a region causes
   * the relevant part of the graph to become more active.
   */
  async absorb(input: string | VectorLike, options: PulseOptions = {}): Promise<Map<string, number>> {
    const scores = await this.pulse(input, options)
    const entries: Array<{ id: string; amount: number; reason: string }> = []
    for (const [id, score] of scores) {
      if (score >= this.config.absorbThreshold) {
        entries.push({ id, amount: clamp01(this.config.absorbGain * score), reason: 'pulse' })
      }
    }
    this.reinforceAll(entries)
    return scores
  }

  // ── Working memory ──

  /**
   * The current "mental state": loaded nodes ranked by `effective` activation
   * (durable decayed score + transient attention) descending, top `limit`.
   */
  workingMemory(limit = 10, minScore = 0): PolyNode[] {
    const withScore: Array<{ node: PolyNode; score: number }> = []
    for (const node of this.graph.query().toArray()) {
      const score = this.effective(node.id)
      if (score > minScore) withScore.push({ node, score })
    }
    withScore.sort((a, b) => b.score - a.score)
    return withScore.slice(0, limit).map(entry => entry.node)
  }
}

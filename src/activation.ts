import { Subscription } from 'rxjs'
import type { PolyGraph } from './graph.js'
import type { PolyNode, NodeActivation, MemoryClass } from './types.js'
import { ACTIVATION_DEFAULTS, clamp01, decayActivationState, decayFactor } from './utils.js'
import { cosineSimilarity } from './vector-index.js'

const HOUR = 3_600_000
const DAY = 24 * HOUR

/** Per-class half-life pair for the score/importance decay curves. */
export interface ClassHalfLives {
  scoreHalfLifeMs?: number
  importanceHalfLifeMs?: number
}

/**
 * Default per-class decay curves. Episodic memories fade fastest unless
 * reinforced; semantic/procedural facts are far more durable; entities barely
 * decay at all. Only used for nodes whose resolved `memoryClass` (see
 * {@link PolyNode.memoryClass} / `NodeTypeDefinition.memoryClass`) has no
 * explicit override in `ActivationConfig.classHalfLives`.
 */
const DEFAULT_CLASS_HALF_LIVES: Record<MemoryClass, Required<ClassHalfLives>> = {
  episodic: { scoreHalfLifeMs: 12 * HOUR, importanceHalfLifeMs: 7 * DAY },
  semantic: { scoreHalfLifeMs: 7 * DAY, importanceHalfLifeMs: 90 * DAY },
  procedural: { scoreHalfLifeMs: 7 * DAY, importanceHalfLifeMs: 60 * DAY },
  entity: { scoreHalfLifeMs: 30 * DAY, importanceHalfLifeMs: Infinity },
}

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
  classHalfLives: DEFAULT_CLASS_HALF_LIVES,
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
  /**
   * Per-memory-class overrides for the score/importance half-lives, keyed by
   * `MemoryClass`. A node's class resolves as `node.memoryClass ??
   * graph.nodeTypes.get(node.type)?.memoryClass`; unclassified nodes always
   * use `scoreHalfLifeMs`/`importanceHalfLifeMs` above, unaffected by this map.
   * Defaults to {@link DEFAULT_CLASS_HALF_LIVES}. Pass `{}` to disable
   * class-based differentiation entirely.
   */
  classHalfLives?: Partial<Record<MemoryClass, ClassHalfLives>>
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
  /** When given, `absorb` reinforces this context (in addition to global score) on every absorbed node. */
  context?: string
}

export type VectorLike = number[] | Float32Array | Float64Array

export interface WorkingMemoryOptions {
  /** Maximum nodes to return. Default 10. */
  limit?: number
  /** Discard candidates whose `effective` score is at or below this. Default 0. */
  minScore?: number
  /** Rank/select within this context (see {@link ActivationEngine.effective}) instead of the global score. */
  context?: string
  /** When true, use global activation for nodes with no score in `context`. Default false. */
  contextFallback?: boolean
  /** Sum of `costOf(node)` across selected nodes must not exceed this. Default: unbounded. */
  tokenBudget?: number
  /** Per-node cost against `tokenBudget`. Defaults to {@link estimateNodeTokens}. */
  costOf?: (node: PolyNode) => number
  /** MMR trade-off: 0 = pure relevance (default), 1 = pure diversity. */
  diversityLambda?: number
  /** Redundancy metric between two candidate nodes. Default: cosine similarity of `node.vector` (0 if either lacks one). */
  similarityOf?: (a: PolyNode, b: PolyNode) => number
}

/** Explainable components of a composite semantic pulse score. */
export interface ScoreBreakdown {
  semantic: number
  graph: number
  recency: number
  usage: number
  weightedSemantic: number
  weightedGraph: number
  weightedRecency: number
  weightedUsage: number
  total: number
}

/** Conservative token estimate for a node when no model-specific tokenizer is available. */
export function estimateNodeTokens(node: PolyNode): number {
  const content = JSON.stringify({ id: node.id, type: node.type, data: node.data })
  return Math.max(1, Math.ceil((content?.length ?? 0) / 4))
}

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
  const result: NodeActivation = {
    score: Math.max(ex.score, inc.score),
    importance: Math.max(ex.importance, inc.importance),
    reinforcementCount: Math.max(existing.reinforcementCount, incoming.reinforcementCount),
    lastMeaningfulActivation: now,
  }
  if (ex.inhibition !== undefined || inc.inhibition !== undefined) {
    result.inhibition = Math.max(ex.inhibition ?? 0, inc.inhibition ?? 0)
    result.lastInhibitedAt = now
  }
  if (ex.context || inc.context) {
    const context: Record<string, { score: number; lastMeaningfulActivation: number }> = {}
    for (const key of new Set([...Object.keys(ex.context ?? {}), ...Object.keys(inc.context ?? {})])) {
      const a = ex.context?.[key]
      const b = inc.context?.[key]
      context[key] = { score: Math.max(a?.score ?? 0, b?.score ?? 0), lastMeaningfulActivation: now }
    }
    result.context = context
  }
  return result
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
  readonly classHalfLives: Partial<Record<MemoryClass, ClassHalfLives>>

  private attention = new Map<string, number>()
  /** Per-node signal breakdown from the most recent `scoreOf` call (via `pulse`), consumed by `recordFeedback`. */
  private lastSignals = new Map<string, { semantic: number; graph: number; recency: number; usage: number }>()
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
    this.classHalfLives = config.classHalfLives ?? DEFAULT_CONFIG.classHalfLives
    this.subscription = graph.changes.subscribe((event) => {
      if (event.type === 'node_removed' && event.nodeId) {
        this.attention.delete(event.nodeId)
        this.lastSignals.delete(event.nodeId)
      }
    })
  }

  /** Stop observing the graph and drop transient attention. Durable state is untouched. */
  dispose(): void {
    this.subscription.unsubscribe()
    this.attention.clear()
    this.lastSignals.clear()
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

  /**
   * Resolve a node's effective score/importance half-lives: `node.memoryClass
   * ?? graph.nodeTypes.get(node.type)?.memoryClass` picks the class, then
   * `classHalfLives[class]` overrides `config.scoreHalfLifeMs`/
   * `importanceHalfLifeMs` per-field. A node with no resolvable class (or a
   * class with no configured override) uses the flat config defaults
   * unchanged — this never changes existing, unclassified nodes' decay.
   */
  resolveHalfLives(node: PolyNode): { scoreHalfLifeMs: number; importanceHalfLifeMs: number } {
    const memoryClass = node.memoryClass ?? this.graph.nodeTypes.get(node.type)?.memoryClass
    const override = memoryClass ? this.classHalfLives[memoryClass] : undefined
    return {
      scoreHalfLifeMs: override?.scoreHalfLifeMs ?? this.config.scoreHalfLifeMs,
      importanceHalfLifeMs: override?.importanceHalfLifeMs ?? this.config.importanceHalfLifeMs,
    }
  }

  /**
   * Durable decayed score plus any transient attention, minus decayed
   * inhibition (clamped to [0,1]). When `context` is given, the context-scoped
   * score is used instead of the global score (0 if the node has no history in
   * that context) — global and context are different lenses, not blended.
   * Inhibition is applied only here, at the final read/ranking layer, never
   * inside `pulse`'s composite — so a suppressed node stays re-evaluable.
   * Decay uses the node's resolved memory-class half-lives (see
   * {@link resolveHalfLives}) when it has one, else the flat config defaults.
   */
  effective(id: string, context?: string): number {
    const node = this.graph.getNode(id)
    if (!node) return 0
    return this.effectiveNode(node, context)
  }

  private effectiveNode(node: PolyNode, context?: string, contextFallback = false): number {
    if (!node.activation) return this.attentionOf(node.id)
    // Decay the raw stored record exactly once, with the resolved half-life —
    // NOT `graph.getActivationState(id)`, which already decay-corrects with
    // the flat default and would double-decay a class-resolved half-life.
    const halfLives = this.resolveHalfLives(node)
    const decayed = decayActivationState(node.activation, Date.now(), halfLives.scoreHalfLifeMs, halfLives.importanceHalfLifeMs)
    const contextScore = context !== undefined ? decayed.context?.[context]?.score : undefined
    const base = context !== undefined && (contextScore !== undefined || !contextFallback) ? (contextScore ?? 0) : decayed.score
    return clamp01(base + this.attentionOf(node.id) - (decayed.inhibition ?? 0))
  }

  /** Current decayed inhibition for `id` (0 when none). */
  inhibitionOf(id: string): number {
    return this.graph.getActivationState(id)?.inhibition ?? 0
  }

  // ── Durable reinforcement (persisted and synced additively) ──

  /** Reinforce a loaded node's durable activation. See {@link PolyGraph.reinforceNode}. */
  reinforce(id: string, amount: number, reason?: string, context?: string): void {
    this.graph.reinforceNode(id, amount, reason, context)
  }

  /** Reinforce several nodes, coalescing change events into one flush. */
  reinforceAll(entries: Array<{ id: string; amount: number; reason?: string; context?: string }>): void {
    if (entries.length === 0) return
    const graph = this.graph
    graph.startBatch()
    try {
      for (const entry of entries) graph.reinforceNode(entry.id, entry.amount, entry.reason, entry.context)
    } finally {
      graph.endBatch()
    }
  }

  /**
   * Suppress a loaded node's durable `inhibition` (mirrors {@link reinforce}
   * but for the inhibition axis). A negative `amount` releases suppression.
   * See {@link PolyGraph.suppressNode}.
   */
  suppress(id: string, amount: number, reason?: string): void {
    this.graph.suppressNode(id, amount, reason)
  }

  // ── Learned weights ──

  /**
   * Record whether `nodeId` — previously scored by `pulse`/`absorb` — turned
   * out useful, nudging the composite `weights` (`semantic`/`graph`/
   * `recency`/`usage`, used by {@link scoreOf}) toward whichever signal was
   * strongest for it: each weight moves by `learningRate * direction *
   * signal`, where `direction` is +1 for useful and -1 for not, so a signal
   * that was high for a useful node gets reinforced and a signal that was
   * high for a useless one gets discounted. Weights are clamped to stay
   * non-negative. A no-op if `nodeId` has no cached signal breakdown (i.e.
   * wasn't scored by a `pulse` call since it was last cleared/scored) — this
   * is a simple exponential-moving-average-style nudge, not a full online
   * learner; `learningRate` defaults to 0.05, matching this codebase's other
   * reinforcement gain constants (`importanceGain`, `absorbGain`). Weights
   * are in-memory only — not persisted or synced.
   */
  recordFeedback(nodeId: string, wasUseful: boolean, learningRate = 0.05): void {
    const signals = this.lastSignals.get(nodeId)
    if (!signals) return
    const direction = wasUseful ? 1 : -1
    this.weights.semantic = Math.max(0, this.weights.semantic + learningRate * direction * signals.semantic)
    this.weights.graph = Math.max(0, this.weights.graph + learningRate * direction * signals.graph)
    this.weights.recency = Math.max(0, this.weights.recency + learningRate * direction * signals.recency)
    this.weights.usage = Math.max(0, this.weights.usage + learningRate * direction * signals.usage)
  }

  /** Return a defensive copy of the learned pulse weights for persistence by callers. */
  getWeights(): Required<NonNullable<ActivationConfig['weights']>> {
    return { ...this.weights }
  }

  /** Restore learned pulse weights supplied by a caller after loading persisted configuration. */
  setWeights(weights: NonNullable<ActivationConfig['weights']>): void {
    for (const key of ['semantic', 'graph', 'recency', 'usage'] as const) {
      const value = weights[key]
      if (value !== undefined) {
        if (!Number.isFinite(value) || value < 0) throw new RangeError(`weights.${key} must be finite and non-negative`)
        this.weights[key] = value
      }
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
    return this.scoreBreakdownOf(node, semantic, graphContribution, now).total
  }

  /** Return the signal and weighted contribution breakdown used for a composite score. */
  scoreBreakdownOf(node: PolyNode, semantic: number, graphContribution: number, now = Date.now()): ScoreBreakdown {
    const recency = decayFactor(now - node.insertedAt, this.config.recencyHalfLifeMs)
    const usage = this.usageOf(node)
    const breakdown: ScoreBreakdown = {
      semantic,
      graph: graphContribution,
      recency,
      usage,
      weightedSemantic: this.weights.semantic * semantic,
      weightedGraph: this.weights.graph * graphContribution,
      weightedRecency: this.weights.recency * recency,
      weightedUsage: this.weights.usage * usage,
      total: 0,
    }
    breakdown.total = breakdown.weightedSemantic + breakdown.weightedGraph + breakdown.weightedRecency + breakdown.weightedUsage
    this.lastSignals.set(node.id, { semantic, graph: graphContribution, recency, usage })
    return breakdown
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
    const entries: Array<{ id: string; amount: number; reason: string; context?: string }> = []
    for (const [id, score] of scores) {
      if (score >= this.config.absorbThreshold) {
        entries.push({ id, amount: clamp01(this.config.absorbGain * score), reason: 'pulse', context: options.context })
      }
    }
    this.reinforceAll(entries)
    return scores
  }

  // ── Working memory ──

  /**
   * The current "mental state": loaded nodes ranked by `effective` activation
   * (durable decayed score + transient attention, minus inhibition) descending,
   * top `limit`.
   *
   * Passing an options object instead enables a budgeted, diversity-aware
   * selection — a memory-flavoured maximal-marginal-relevance pass suited to
   * LLM context assembly: greedily picks the highest `relevance − λ ×
   * similarity-to-already-selected` candidate, under a token budget, so the
   * result isn't 20 near-duplicate highly-activated neighbours.
   */
  workingMemory(limit?: number, minScore?: number): PolyNode[]
  workingMemory(options: WorkingMemoryOptions): PolyNode[]
  workingMemory(limitOrOptions: number | WorkingMemoryOptions = 10, minScoreArg = 0): PolyNode[] {
    const options: WorkingMemoryOptions =
      typeof limitOrOptions === 'number' ? { limit: limitOrOptions, minScore: minScoreArg } : limitOrOptions

    const minScore = options.minScore ?? 0
    const candidates: Array<{ node: PolyNode; score: number }> = []
    for (const node of this.graph.query().toArray()) {
      const score = this.effectiveNode(node, options.context, options.contextFallback)
      if (score > minScore) candidates.push({ node, score })
    }
    return this.selectWorkingMemory(candidates, options)
  }

  /** Rank persisted nodes without requiring them to be loaded into the hot cache. */
  async workingMemoryPersisted(options: WorkingMemoryOptions = {}): Promise<PolyNode[]> {
    const minScore = options.minScore ?? 0
    const candidates: Array<{ node: PolyNode; score: number }> = []
    for (const node of await this.graph.queryPersisted().toArray()) {
      const score = this.effectiveNode(node, options.context, options.contextFallback)
      if (score > minScore) candidates.push({ node, score })
    }
    return this.selectWorkingMemory(candidates, options)
  }

  private selectWorkingMemory(candidates: Array<{ node: PolyNode; score: number }>, options: WorkingMemoryOptions): PolyNode[] {
    const limit = options.limit ?? 10
    const costOf = options.costOf ?? estimateNodeTokens
    const diversityLambda = options.diversityLambda ?? 0
    const similarityOf = options.similarityOf ?? defaultSimilarity
    const budget = options.tokenBudget ?? Infinity
    candidates.sort((a, b) => b.score - a.score)

    if (diversityLambda <= 0 && budget === Infinity) {
      return candidates.slice(0, limit).map(entry => entry.node)
    }

    // Cap the candidate pool before the O(pool * selected) MMR pass, so a
    // graph with many loaded nodes doesn't turn selection quadratic.
    const pool = candidates.slice(0, Math.max(limit * 4, limit))

    const selected: PolyNode[] = []
    const remaining = [...pool]
    let spent = 0
    while (selected.length < limit && remaining.length > 0) {
      let bestIndex = -1
      let bestScore = -Infinity
      for (let i = 0; i < remaining.length; i++) {
        const candidate = remaining[i]
        const maxSimilarity = selected.length === 0
          ? 0
          : Math.max(...selected.map(s => similarityOf(candidate.node, s)))
        const mmr = diversityLambda > 0
          ? (1 - diversityLambda) * candidate.score - diversityLambda * maxSimilarity
          : candidate.score
        if (mmr > bestScore) {
          bestScore = mmr
          bestIndex = i
        }
      }
      const [chosen] = remaining.splice(bestIndex, 1)
      const cost = costOf(chosen.node)
      if (!Number.isFinite(cost) || cost < 0) throw new RangeError('costOf(node) must return a finite non-negative number')
      if (spent + cost > budget) break
      spent += cost
      selected.push(chosen.node)
    }
    return selected
  }
}

/** Cosine similarity between two nodes' vectors, or 0 when either lacks one. */
function defaultSimilarity(a: PolyNode, b: PolyNode): number {
  if (!a.vector || !b.vector) return 0
  return cosineSimilarity(a.vector, b.vector)
}

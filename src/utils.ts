import type { PolyNode, NodeActivation } from './types.js'

const HOUR = 3_600_000
const DAY = 24 * HOUR

/** Yield to the event loop (a zero-delay timer) so pending UI work/renders can run between large synchronous batches. */
export function yieldToUI(): Promise<void> {
  return new Promise(r => setTimeout(r, 0))
}

export function cloneData<T>(value: T): T {
  return structuredClone(value)
}

export function clonePolyNode<T extends PolyNode>(node: T): T {
  return {
    ...node,
    data: cloneData(node.data),
    vector: node.vector ? new Float64Array(node.vector) : undefined,
    activation: node.activation ? { ...node.activation } : undefined,
  }
}

/** Default activation decay curves, shared by {@link PolyGraph} and the `activation` module. */
export const ACTIVATION_DEFAULTS = {
  /** Half-life of the short-term `score` curve. */
  scoreHalfLifeMs: DAY,
  /** Half-life of the long-term `importance` curve. */
  importanceHalfLifeMs: 30 * DAY,
  /** Fraction of a reinforcement delta folded into `importance`. */
  importanceGain: 0.05,
} as const

/** Clamp a number into [0, 1]. */
export function clamp01(value: number): number {
  return value < 0 ? 0 : value > 1 ? 1 : value
}

/**
 * Exponential-decay multiplier for activation: `0.5 ** (elapsed / halfLife)`.
 * Returns 1 for non-positive elapsed times and a non-decaying (infinite or
 * non-positive) half-life. Deterministic and monotonic, so replicas computing
 * current activation from the same anchor and their own clocks converge.
 */
export function decayFactor(elapsedMs: number, halfLifeMs: number): number {
  if (!Number.isFinite(elapsedMs) || elapsedMs <= 0) return 1
  if (!Number.isFinite(halfLifeMs) || halfLifeMs <= 0) return 1
  const factor = Math.pow(0.5, elapsedMs / halfLifeMs)
  return factor < 1 ? factor : 1
}

/**
 * Read-only decay-corrected view of an activation record. Does NOT re-anchor
 * `lastMeaningfulActivation`; the stored values always represent the state at
 * that anchor, so reinforcement/merge can decay-correct and re-anchor exactly.
 */
export function decayActivationState(
  activation: NodeActivation,
  now: number,
  scoreHalfLifeMs = ACTIVATION_DEFAULTS.scoreHalfLifeMs,
  importanceHalfLifeMs = ACTIVATION_DEFAULTS.importanceHalfLifeMs,
): NodeActivation {
  return {
    score: clamp01(activation.score * decayFactor(now - activation.lastMeaningfulActivation, scoreHalfLifeMs)),
    importance: clamp01(activation.importance * decayFactor(now - activation.lastMeaningfulActivation, importanceHalfLifeMs)),
    reinforcementCount: activation.reinforcementCount,
    lastMeaningfulActivation: activation.lastMeaningfulActivation,
  }
}

/**
 * Apply a reinforcement delta to an activation record: decay-correct the prior
 * state to `now`, add `delta` to `score`, fold a fraction (`importanceGain`)
 * into `importance`, and re-anchor `lastMeaningfulActivation` to `now` so the
 * stored record again represents "value at the anchor".
 */
export function reinforceActivation(
  previous: NodeActivation | undefined,
  delta: number,
  now: number,
  halfLives: { scoreHalfLifeMs: number; importanceHalfLifeMs: number; importanceGain: number } = ACTIVATION_DEFAULTS,
): NodeActivation {
  const decayed = previous
    ? decayActivationState(previous, now, halfLives.scoreHalfLifeMs, halfLives.importanceHalfLifeMs)
    : { score: 0, importance: 0 }
  return {
    score: clamp01(decayed.score + delta),
    importance: clamp01(decayed.importance + halfLives.importanceGain * delta),
    reinforcementCount: (previous?.reinforcementCount ?? 0) + 1,
    lastMeaningfulActivation: now,
  }
}

/** Current decayed activation score of a node (0 when it has none). */
export function activationScoreOf(
  node: PolyNode,
  now = Date.now(),
  halfLifeMs = ACTIVATION_DEFAULTS.scoreHalfLifeMs,
): number {
  if (!node.activation) return 0
  return clamp01(node.activation.score * decayFactor(now - node.activation.lastMeaningfulActivation, halfLifeMs))
}

export function assertNonNegativeInteger(value: number, name: string): void {
  if (!Number.isInteger(value) || value < 0) {
    throw new RangeError(`${name} must be a non-negative integer`)
  }
}

export function assertFiniteVector(vector: ArrayLike<number>, name = 'vector'): void {
  for (let index = 0; index < vector.length; index++) {
    if (!Number.isFinite(vector[index])) {
      throw new RangeError(`${name} must contain only finite numbers`)
    }
  }
}

/** Persistence key for an edge. `source` and `type` must not contain `::` (the separator); `target` may. */
export function edgeId(source: string, type: string, target: string): string {
  if (source.includes('::') || type.includes('::')) {
    throw new RangeError('Edge source and type must not contain "::"')
  }
  return `${source}::${type}::${target}`
}

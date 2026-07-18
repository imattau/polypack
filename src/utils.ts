import type { PolyNode } from './types.js'

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
  }
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

export function edgeId(source: string, type: string, target: string): string {
  if (source.includes('::') || type.includes('::')) {
    throw new RangeError('Edge source and type must not contain "::"')
  }
  return `${source}::${type}::${target}`
}

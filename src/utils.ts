export function yieldToUI(): Promise<void> {
  return new Promise(r => setTimeout(r, 0))
}

export function edgeId(source: string, type: string, target: string): string {
  if (source.includes('::') || type.includes('::')) {
    throw new RangeError('Edge source and type must not contain "::"')
  }
  return `${source}::${type}::${target}`
}

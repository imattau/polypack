export function yieldToUI(): Promise<void> {
  return new Promise(r => setTimeout(r, 0))
}

export function edgeId(source: string, type: string, target: string): string {
  return `${source}::${type}::${target}`
}

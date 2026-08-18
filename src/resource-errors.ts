export class ResourceLimitError extends Error {
  readonly name = 'ResourceLimitError'
  readonly limitName: string
  readonly limit: number

  constructor(limitName: string, limit: number) {
    super(`Resource exceeded ${limitName} limit of ${limit}`)
    this.limitName = limitName
    this.limit = limit
  }
}

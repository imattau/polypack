export class QueryLimitError extends Error {
  readonly name = 'QueryLimitError'
  readonly limitName: string
  readonly limit: number

  constructor(limitName: string, limit: number) {
    super(`Query exceeded ${limitName} limit of ${limit}`)
    this.limitName = limitName
    this.limit = limit
  }
}

export class QueryAbortedError extends Error {
  readonly name = 'QueryAbortedError'

  constructor() {
    super('Query was aborted')
  }
}

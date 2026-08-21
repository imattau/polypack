export class SchemaValidationError extends Error {
  readonly name = 'SchemaValidationError'
  readonly kind: 'node' | 'edge'
  readonly type: string

  constructor(kind: 'node' | 'edge', type: string, message: string) {
    super(`${kind} type ${type} failed validation: ${message}`)
    this.kind = kind
    this.type = type
  }
}

export class AdapterCapabilityError extends Error {
  readonly name = 'AdapterCapabilityError'
  readonly capability: string

  constructor(capability: string) {
    super(`Persistence adapter does not support capability: ${capability}`)
    this.capability = capability
  }
}

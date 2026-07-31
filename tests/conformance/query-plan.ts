/** TypeScript view of specification/query-plan.schema.json. */
export interface QueryPlan {
  nodeTypes?: string[]
  attributes?: Array<
    | { field: string; operator: 'eq'; value: unknown }
    | { field: string; operator: 'range'; above?: number; below?: number }
  >
  edgeFilter?: { type: string; target?: string; source?: string }
  traversal?: Array<{ edgeType: string; direction: 'out' | 'in'; depth: number }>
  joins?: Array<{ edgeType: string; direction: 'out' | 'in' }>
  similarity?: { vector: number[]; threshold: number; topK?: number }
  order?: { field: string; direction: 'asc' | 'desc' }
  offset?: number
  limit?: number
}

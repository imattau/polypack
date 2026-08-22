/**
 * Example: local-first personal knowledge memory.
 *
 * This is the reference workflow for an MCP-backed memory tool:
 * ingest durable facts with provenance, retrieve a semantic neighbourhood,
 * promote the useful part into context-specific working memory, and preserve
 * history when facts are corrected or consolidated.
 *
 * Run: npx tsx examples/personal-memory.ts
 */

import { ActivationEngine, FeatureHashEmbedding, MemoryAdapter, PolyGraph } from '../src/index'
import type { PolyNode } from '../src/index'

export interface PersonalMemory {
  graph: PolyGraph
  engine: ActivationEngine
}

export function createPersonalMemory(): PersonalMemory {
  const graph = new PolyGraph(
    new MemoryAdapter(),
    undefined,
    new FeatureHashEmbedding({ dimensions: 256 }),
  )
  return { graph, engine: new ActivationEngine(graph) }
}

export async function remember(
  memory: PersonalMemory,
  id: string,
  text: string,
  options: {
    type?: string
    memoryClass?: PolyNode['memoryClass']
    confidence?: number
    source?: string
    observedAt?: number
  } = {},
): Promise<void> {
  const now = Date.now()
  await memory.graph.addNodeWithEmbedding({
    id,
    type: options.type ?? 'memory',
    data: { text },
    memoryClass: options.memoryClass ?? 'episodic',
    confidence: options.confidence,
    source: options.source ?? 'user',
    observedAt: options.observedAt ?? now,
    insertedAt: now,
    updatedAt: now,
  }, text)
}

export async function recall(
  memory: PersonalMemory,
  query: string,
  context: string,
  tokenBudget = 180,
): Promise<PolyNode[]> {
  // `absorb` is the durable equivalent of an MCP read: relevant nodes gain
  // activation in this context and can be recalled in a later turn.
  await memory.engine.absorb(query, { context, topK: 8, semanticThreshold: 0.05 })
  return memory.engine.workingMemory({
    context,
    contextFallback: true,
    limit: 8,
    tokenBudget,
    diversityLambda: 0.2,
  })
}

async function main(): Promise<void> {
  const memory = createPersonalMemory()

  await remember(memory, 'fact:polypack', 'Polypack combines a property graph, vector search, and adaptive memory.', {
    type: 'fact', memoryClass: 'semantic', confidence: 0.98, source: 'project:README',
  })
  await remember(memory, 'note:architecture', 'The local-first design keeps the working set on-device and syncs durable changes.', {
    type: 'note', source: 'user',
  })
  await remember(memory, 'fact:old-sync', 'The sync layer uses last-write-wins for every memory update.', {
    type: 'fact', memoryClass: 'semantic', confidence: 0.55, source: 'user:2026-08-01',
  })
  await remember(memory, 'fact:new-sync', 'The sync layer deduplicates operations and merges activation updates additively.', {
    type: 'fact', memoryClass: 'semantic', confidence: 0.95, source: 'project:CHANGELOG',
  })
  memory.graph.supersede('fact:new-sync', 'fact:old-sync', 1, 'corrected-by-newer-evidence')

  memory.graph.addEdge('fact:polypack', 'RELATES_TO', 'note:architecture')
  memory.graph.addEdge('fact:new-sync', 'RELATES_TO', 'note:architecture')

  const results = await recall(memory, 'How does Polypack keep relevant knowledge local and current?', 'project:polypack')
  console.log('── Personal memory recall ──')
  for (const node of results) {
    const confidence = node.confidence === undefined ? 'n/a' : node.confidence.toFixed(2)
    console.log(`  ${node.id}  confidence=${confidence}  ${(node.data as { text: string }).text}`)
  }

  const summary = memory.graph.consolidate({
    id: 'summary:polypack-memory',
    type: 'summary',
    data: { text: 'Polypack keeps relevant, sourced knowledge local while preserving corrected history.' },
    memoryClass: 'semantic',
    confidence: 0.93,
    source: 'inference:personal-memory',
    insertedAt: Date.now(),
    updatedAt: Date.now(),
  }, ['fact:polypack', 'note:architecture'], { reason: 'consolidated-for-context' })
  if (!summary) throw new Error('failed to consolidate memory')
  await memory.graph.flush()
  console.log(`\n✓ ${memory.graph.size} memories retained; summary derived from ${summary.derivedFrom?.length ?? 0} sources`)
  memory.engine.dispose()
  await memory.graph.dispose()
}

if (process.argv[1]?.endsWith('personal-memory.ts')) await main()


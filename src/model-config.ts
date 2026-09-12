import type { ClaudeCatalogModel } from './types.ts'

export const BASE_BETAS: readonly string[] = [
  'claude-code-20250219',
  'oauth-2025-04-20',
  'interleaved-thinking-2025-05-14',
  'prompt-caching-scope-2026-01-05',
  'context-management-2025-06-27',
  'advisor-tool-2026-03-01',
  'thinking-token-count-2026-05-13',
  'extended-cache-ttl-2025-04-11',
]

export const LONG_CONTEXT_BETAS: readonly string[] = [
  'context-1m-2025-08-07',
  'interleaved-thinking-2025-05-14',
]

const EFFORT_BETA = 'effort-2025-11-24'
const ALL_EFFORTS: readonly string[] = ['low', 'medium', 'high', 'xhigh', 'max']

interface ModelOverride {
  exclude?: string[]
  add?: string[]
  disableEffort?: boolean
  /** Explicit effort ladder override; wins over EFFORT_EFFORTS inference. */
  efforts?: string[]
}

const MODEL_OVERRIDES: ReadonlyArray<[string, ModelOverride]> = [
  ['fable', { add: [EFFORT_BETA] }],
  ['opus-4-5', { add: [EFFORT_BETA], efforts: ['low', 'medium', 'high'] }],
  ['opus-4-6', { add: [EFFORT_BETA], efforts: ['low', 'medium', 'high', 'max'] }],
  ['opus-4-7', { add: [EFFORT_BETA], efforts: ['low', 'medium', 'high', 'xhigh', 'max'] }],
  ['opus-4-8', { add: [EFFORT_BETA], efforts: ['low', 'medium', 'high', 'xhigh', 'max'] }],
  ['opus-5', { add: [EFFORT_BETA], efforts: ['low', 'medium', 'high', 'xhigh', 'max'] }],
  ['sonnet-4-5', { add: [EFFORT_BETA], efforts: ['low', 'medium', 'high'] }],
  ['sonnet-4-6', { add: [EFFORT_BETA], efforts: ['low', 'medium', 'high', 'max'] }],
  ['sonnet-4-7', { add: [EFFORT_BETA] }],
  ['sonnet-4-8', { add: [EFFORT_BETA] }],
  ['sonnet-5', { add: [EFFORT_BETA], efforts: ['low', 'medium', 'high', 'xhigh', 'max'] }],
  ['haiku', { disableEffort: true }],
]

function getModelOverride(modelId: string): ModelOverride | undefined {
  const lower = modelId.toLowerCase()
  for (const [pattern, override] of MODEL_OVERRIDES) {
    if (lower.includes(pattern)) return override
  }
  return undefined
}

export function effortLadder(modelId: string): string[] | undefined {
  const override = getModelOverride(modelId)
  if (override?.disableEffort) return undefined
  if (override?.efforts !== undefined) return [...override.efforts]
  if (override?.add?.includes(EFFORT_BETA)) return [...ALL_EFFORTS]
  return undefined
}

export function getModelBetas(modelId: string, excluded: ReadonlySet<string>): string[] {
  let betas = [...BASE_BETAS]
  const override = getModelOverride(modelId)
  if (override) {
    if (override.exclude) betas = betas.filter(beta => !override.exclude!.includes(beta))
    if (override.add) for (const beta of override.add) if (!betas.includes(beta)) betas.push(beta)
  }
  return betas.filter(beta => !excluded.has(beta))
}

export function supportsEffort(modelId: string): boolean {
  return effortLadder(modelId) !== undefined
}

export function isLongContextError(responseBody: string): boolean {
  return responseBody.includes('Extra usage is required for long context requests')
    || responseBody.includes('long context beta is not yet available')
    || responseBody.includes('You\'re out of extra usage')
}

export function getNextBetaToExclude(modelId: string, excluded: ReadonlySet<string>): string | undefined {
  for (const beta of LONG_CONTEXT_BETAS) {
    if (!excluded.has(beta)) return beta
  }
  return undefined
}

export const DEFAULT_CATALOG: readonly ClaudeCatalogModel[] = [
  {
    id: 'claude-sonnet-5',
    name: 'Claude Sonnet 5',
    description: 'Best speed-to-intelligence balance (default).',
    contextWindow: 1_000_000,
    maxTokens: 128_000,
    reasoningEfforts: ['low', 'medium', 'high', 'xhigh', 'max'],
  },
  {
    id: 'claude-opus-5',
    name: 'Claude Opus 5',
    description: 'Complex agentic coding and deep reasoning.',
    contextWindow: 1_000_000,
    maxTokens: 128_000,
    reasoningEfforts: ['low', 'medium', 'high', 'xhigh', 'max'],
  },
  {
    id: 'claude-haiku-4-5',
    name: 'Claude Haiku 4.5',
    description: 'Fastest model, near-frontier intelligence.',
    contextWindow: 200_000,
    maxTokens: 64_000,
  },
  {
    id: 'claude-fable-5-1',
    name: 'Claude Fable 5.1',
    description: 'Demanded reasoning and long-horizon agentic work.',
    contextWindow: 1_000_000,
    maxTokens: 128_000,
    reasoningEfforts: ['low', 'medium', 'high', 'xhigh', 'max'],
  },
]


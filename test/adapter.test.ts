import { describe, expect, test } from 'bun:test'
import { createUserMessage, type GenerateOptions } from '@deepseek-ai/dsh-llm'
import { ClaudeSubscriptionAdapter, PROVIDER_ID, PROVIDER_NAME } from '../src/adapter.ts'
import { DEFAULT_CATALOG } from '../src/model-config.ts'
import type { ClaudeConnectionOptions } from '../src/types.ts'
import type { ResolvedCredential } from '../src/auth.ts'

const CONN: ClaudeConnectionOptions = {
  credentialsPath: '',
  disabledModels: [],
  streamIdleTimeoutMs: 60_000,
  requestTimeoutMs: 120_000,
}

const CRED: ResolvedCredential = { accessToken: 'tok', source: 'test' }

function makeAdapter(fetchImpl: typeof fetch): ClaudeSubscriptionAdapter {
  return new ClaudeSubscriptionAdapter({
    resolveConnection: () => CONN,
    resolveCredential: async () => CRED,
    resolveModels: async () => [...DEFAULT_CATALOG],
    fetchImpl,
  })
}

function textBody(): string {
  return [
    'event: message_start',
    'data: {"type":"message_start","message":{"id":"m1","model":"claude-sonnet-5","usage":{"input_tokens":10,"cache_read_input_tokens":2,"output_tokens":1}}}',
    '',
    'event: content_block_start',
    'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}',
    '',
    'event: content_block_delta',
    'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi there"}}',
    '',
    'event: content_block_stop',
    'data: {"type":"content_block_stop","index":0}',
    '',
    'event: message_delta',
    'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":7}}',
    '',
    'event: message_stop',
    'data: {"type":"message_stop"}',
    '',
    '',
  ].join('\n')
}

function makeRequest(overrides: Partial<GenerateOptions> = {}): GenerateOptions {
  return {
    provider: PROVIDER_ID,
    model: 'claude-sonnet-5',
    messages: [createUserMessage({ content: [{ type: 'text', text: 'hello' }], source: { kind: 'user' as const, user: 'U' } })],
    maxTokens: 1000,
    ...overrides,
  }
}

describe('ClaudeSubscriptionAdapter', () => {
  test('streams a text response into harness chunks', async () => {
    const fetchImpl = (async () => new Response(textBody(), {
      status: 200,
      headers: { 'content-type': 'text/event-stream' },
    })) as unknown as typeof fetch
    const adapter = makeAdapter(fetchImpl)
    const chunks = []
    for await (const chunk of adapter.stream(makeRequest())) chunks.push(chunk)

    expect(chunks[0]).toEqual({ type: 'block-start', index: 0, blockType: 'text' })
    expect(chunks[1]).toEqual({ type: 'text-delta', index: 0, text: 'Hi there' })
    const usage = chunks.find(c => c.type === 'usage') as { usage: { inputTokens: number; outputTokens: number } }
    expect(usage.usage.inputTokens).toBe(8)
    expect(usage.usage.outputTokens).toBe(7)
    expect(chunks[chunks.length - 1]).toEqual({ type: 'finish', reason: { kind: 'stop' } })
  })

  test('sends the first-party header set', async () => {
    let capturedHeaders: Headers | undefined
    let capturedBody: unknown
    const fetchImpl = (async (_url: string, init: RequestInit) => {
      capturedHeaders = new Headers(init.headers)
      capturedBody = JSON.parse(String(init.body))
      return new Response(textBody(), { status: 200 })
    }) as unknown as typeof fetch
    const adapter = makeAdapter(fetchImpl)
    for await (const _ of adapter.stream(makeRequest())) void _

    expect(capturedHeaders?.get('authorization')).toBe('Bearer tok')
    expect(capturedHeaders?.get('anthropic-version')).toBe('2023-06-01')
    expect(capturedHeaders?.get('anthropic-beta')).toContain('claude-code-20250219')
    expect(capturedHeaders?.get('anthropic-beta')).toContain('oauth-2025-04-20')
    expect(capturedHeaders?.get('x-app')).toBe('cli')
    expect(capturedHeaders?.get('user-agent')).toMatch(/^claude-cli\//)
    expect(capturedHeaders?.get('anthropic-dangerous-direct-browser-access')).toBe('true')
    const body = capturedBody as Record<string, unknown>
    expect(body.model).toBe('claude-sonnet-5')
    expect(body.stream).toBe(true)
    expect(Array.isArray(body.system)).toBe(true)
    expect((body.system as Array<{ text: string }>)[1].text).toMatch(/Claude Code/)
    expect((body.system as Array<{ text: string }>)[0].text).toMatch(/^x-anthropic-billing-header/)
  })

  test('maps HTTP error bodies to typed LlmError codes', async () => {
    const fetchImpl = (async () => new Response(JSON.stringify({
      error: { type: 'overloaded_error', message: 'Overloaded' },
    }), { status: 529, headers: { 'content-type': 'application/json' } })) as unknown as typeof fetch
    const adapter = makeAdapter(fetchImpl)
    let error: unknown
    try {
      for await (const _ of adapter.stream(makeRequest())) void _
    } catch (err) {
      error = err
    }
    expect(error).toBeInstanceOf(Error)
    expect((error as { code?: string }).code).toBe('SERVER')
    expect((error as Error).message).toBe('Overloaded')
  })

  test('streams fail with a typed MISSING_CREDENTIAL error when the credential is unavailable', async () => {
    const adapter = new ClaudeSubscriptionAdapter({
      resolveConnection: () => CONN,
      resolveCredential: async () => ({ kind: 'unavailable' as const, reason: 'no Claude Code credentials file at C:\\fake\\.credentials.json — run `claude` once to authenticate' }),
      resolveModels: async () => [...DEFAULT_CATALOG],
    })
    let error: unknown
    try {
      for await (const _ of adapter.stream(makeRequest())) void _
    } catch (err) {
      error = err
    }
    expect(error).toBeInstanceOf(Error)
    expect((error as { code?: string }).code).toBe('MISSING_CREDENTIAL')
    expect((error as Error).message).toMatch(/no Claude Code credentials file/)
  })

  test('401 maps to AUTH', async () => {
    const fetchImpl = (async () => new Response(JSON.stringify({
      error: { type: 'authentication_error', message: 'invalid x-api-key' },
    }), { status: 401 })) as unknown as typeof fetch
    const adapter = makeAdapter(fetchImpl)
    let error: unknown
    try {
      for await (const _ of adapter.stream(makeRequest())) void _
    } catch (err) {
      error = err
    }
    expect((error as { code?: string }).code).toBe('AUTH')
  })

  test('providerInfo and listModels describe the route', async () => {
    const adapter = makeAdapter(((async () => new Response('')) as unknown) as typeof fetch)
    const info = adapter.providerInfo(PROVIDER_ID)
    expect(info).toEqual({ id: PROVIDER_ID, name: PROVIDER_NAME })
    const models = await adapter.listModels(PROVIDER_ID)
    expect(models.map(m => m.id)).toEqual(['claude-sonnet-5', 'claude-opus-5', 'claude-haiku-4-5', 'claude-fable-5-1'])
    const resolved = await adapter.resolveModel(PROVIDER_ID, 'claude-sonnet-5')
    expect(resolved.context).toEqual({ contextWindow: 1_000_000 })
    expect(resolved.reasoning?.efforts.map(e => String(e.id))).toEqual(['low', 'medium', 'high', 'xhigh', 'max'])
    expect(resolved.reasoning?.defaultEffort).toBeUndefined()
  })

  test('disabled models drop out of the catalog', async () => {
    const adapter = new ClaudeSubscriptionAdapter({
      resolveConnection: () => CONN,
      resolveCredential: async () => CRED,
      resolveModels: async () => DEFAULT_CATALOG.filter(m => m.id !== 'claude-opus-5'),
    })
    const models = await adapter.listModels(PROVIDER_ID)
    expect(models.map(m => m.id)).toEqual(['claude-sonnet-5', 'claude-haiku-4-5', 'claude-fable-5-1'])
  })
})

  test('discovered model without catalog efforts still exposes effort levels', async () => {
    const adapter = new ClaudeSubscriptionAdapter({
      resolveConnection: () => CONN,
      resolveCredential: async () => CRED,
      resolveModels: async () => [{ id: 'claude-opus-4-8', name: 'Claude Opus 4.8', contextWindow: 1_000_000, maxTokens: 128_000 }],
    })
    const resolved = await adapter.resolveModel(PROVIDER_ID, 'claude-opus-4-8')
    expect(resolved.reasoning?.efforts.map(e => String(e.id))).toEqual(['low', 'medium', 'high', 'xhigh', 'max'])
    expect(resolved.reasoning?.defaultEffort).toBeUndefined()
  })

  test('haiku model exposes no reasoning efforts', async () => {
    const adapter = new ClaudeSubscriptionAdapter({
      resolveConnection: () => CONN,
      resolveCredential: async () => CRED,
      resolveModels: async () => [{ id: 'claude-haiku-4-5-20251001', name: 'Claude Haiku 4.5', contextWindow: 200_000, maxTokens: 64_000 }],
    })
    const resolved = await adapter.resolveModel(PROVIDER_ID, 'claude-haiku-4-5-20251001')
    expect(resolved.reasoning).toBeUndefined()
  })

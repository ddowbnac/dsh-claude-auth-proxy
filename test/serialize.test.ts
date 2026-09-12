import { describe, expect, test } from 'bun:test'
import { MessageId, ToolCallId, createUserMessage, createSystemMessage } from '@deepseek-ai/dsh-llm'
import { randomUUID } from 'node:crypto'
import { serializeMessages, serializeRequest } from '../src/serialize.ts'
import { SYSTEM_IDENTITY } from '../src/types.ts'

describe('serializeMessages', () => {
  test('user text passes through', () => {
    const user = createUserMessage({ content: [{ type: 'text', text: 'hello' }], source: { kind: 'user' as const, user: 'U' } })
    const wire = serializeMessages([user], false)
    expect(wire).toEqual([{ role: 'user', content: 'hello' }])
  })

  test('system messages are separated', () => {
    const system = createSystemMessage('be brief', 'test')
    const user = createUserMessage({ content: [{ type: 'text', text: 'hi' }], source: { kind: 'user' as const, user: 'U' } })
    const wire = serializeMessages([system, user], false)
    expect(wire.some(m => m.role === 'system')).toBe(false)
    expect(wire).toEqual([{ role: 'user', content: 'hi' }])
  })

  test('tool results become tool_result blocks in a user message', () => {
    const user = createUserMessage({
      content: [
        { type: 'tool-result', toolCallId: ToolCallId('call_1'), content: [{ type: 'text', text: '42' }] },
      ],
      source: { kind: 'user' as const, user: 'U' },
    })
    const wire = serializeMessages([user], false)
    expect(wire).toEqual([{
      role: 'user',
      content: [{ type: 'tool_result', tool_use_id: 'call_1', content: '42' }],
    }])
  })

  test('empty tool results send a placeholder', () => {
    const user = createUserMessage({
      content: [
        { type: 'tool-result', toolCallId: ToolCallId('call_2'), content: [], isError: true },
      ],
      source: { kind: 'user' as const, user: 'U' },
    })
    const wire = serializeMessages([user], false)
    expect(wire[0]).toEqual({
      role: 'user',
      content: [{ type: 'tool_result', tool_use_id: 'call_2', content: '(no output)', is_error: true }],
    })
  })

  test('assistant tool calls replay as tool_use with parsed input', () => {
    const assistant = {
      role: 'assistant' as const,
      id: MessageId(randomUUID()),
      source: { kind: 'model' as const, provider: 'claude-subscription', model: 'claude-sonnet-5' },
      content: [
        { type: 'tool-call' as const, id: ToolCallId('call_9'), name: 'bash', arguments: '{"cmd":"ls"}' },
      ],
    }
    const wire = serializeMessages([assistant], false)
    expect(wire[0]).toEqual({
      role: 'assistant',
      content: [{ type: 'tool_use', id: 'call_9', name: 'bash', input: { cmd: 'ls' } }],
    })
  })

  test('composite harness tool ids (with |) are sanitized to a legal wire id', () => {
    const composite = 'call_f02fc5e488744d8ea3bd785e|fc_7dc935f3'
    const assistant = {
      role: 'assistant' as const,
      id: MessageId(randomUUID()),
      source: { kind: 'model' as const, provider: 'claude-subscription', model: 'claude-sonnet-5' },
      content: [
        { type: 'tool-call' as const, id: ToolCallId(composite), name: 'bash', arguments: '{"cmd":"ls"}' },
      ],
    }
    const result = createUserMessage({
      content: [
        { type: 'tool-result' as const, toolCallId: ToolCallId(composite), content: [{ type: 'text' as const, text: 'ok' }] },
      ],
      source: { kind: 'user' as const, user: 'U' },
    })
    const wire = serializeMessages([assistant, result], false)
    const toolUse = (wire[0].content as unknown as Array<Record<string, unknown>>).find(b => b.type === 'tool_use')
    const toolResult = (wire[1].content as unknown as Array<Record<string, unknown>>).find(b => b.type === 'tool_result')
    // Both must be legal on the Anthropic wire charset.
    expect(String(toolUse!.id)).toMatch(/^[a-zA-Z0-9_-]+$/)
    expect(String(toolResult!.tool_use_id)).toMatch(/^[a-zA-Z0-9_-]+$/)
    // And they must pair (same wire id) so the API accepts the tool_use/tool_result adjacency.
    expect(toolUse!.id).toBe(toolResult!.tool_use_id)
    expect(String(toolUse!.id)).not.toContain('|')
  })

  test('tool id sanitization is deterministic (same raw id maps identically across the request)', () => {
    const composite = 'call_aaa|fc_bbb'
    const a1 = {
      role: 'assistant' as const, id: MessageId(randomUUID()),
      source: { kind: 'model' as const, provider: 'claude-subscription', model: 'claude-sonnet-5' },
      content: [{ type: 'tool-call' as const, id: ToolCallId(composite), name: 'bash', arguments: '{}' }],
    }
    const r1 = createUserMessage({
      content: [{ type: 'tool-result' as const, toolCallId: ToolCallId(composite), content: [{ type: 'text' as const, text: '1' }] }],
      source: { kind: 'user' as const, user: 'U' },
    })
    const a2 = {
      role: 'assistant' as const, id: MessageId(randomUUID()),
      source: { kind: 'model' as const, provider: 'claude-subscription', model: 'claude-sonnet-5' },
      content: [{ type: 'tool-call' as const, id: ToolCallId(composite), name: 'bash', arguments: '{}' }],
    }
    const r2 = createUserMessage({
      content: [{ type: 'tool-result' as const, toolCallId: ToolCallId(composite), content: [{ type: 'text' as const, text: '2' }] }],
      source: { kind: 'user' as const, user: 'U' },
    })
    const wire = serializeMessages([a1, r1, a2, r2], false)
    const ids = wire
      .flatMap(m => Array.isArray(m.content) ? m.content : [])
      .map(b => (b as any).type === 'tool_use' ? (b as any).id : (b as any).type === 'tool_result' ? (b as any).tool_use_id : null)
      .filter(Boolean)
    expect(new Set(ids).size).toBe(1)
  })

  test('assistant reasoning is never replayed as thinking (signature-less replay 400s)', () => {
    const withTool = {
      role: 'assistant' as const,
      id: MessageId(randomUUID()),
      source: { kind: 'model' as const, provider: 'claude-subscription', model: 'claude-sonnet-5' },
      content: [
        { type: 'reasoning' as const, text: 'hmm' },
        { type: 'tool-call' as const, id: ToolCallId('call_10'), name: 'read', arguments: '{}' },
      ],
    }
    const wire1 = serializeMessages([withTool], false)
    expect(wire1[0].content).toEqual([
      { type: 'tool_use', id: 'call_10', name: 'read', input: {} },
    ])

    const plain = {
      role: 'assistant' as const,
      id: MessageId(randomUUID()),
      source: { kind: 'model' as const, provider: 'claude-subscription', model: 'claude-sonnet-5' },
      content: [
        { type: 'reasoning' as const, text: 'hmm' },
        { type: 'text' as const, text: 'answer' },
      ],
    }
    const wire2 = serializeMessages([plain], false)
    expect(wire2[0].content).toEqual([{ type: 'text', text: 'answer' }])
  })

  test('reasoning-only turn without text or tool call is dropped from the wire', () => {
    const reasoningOnly = {
      role: 'assistant' as const,
      id: MessageId(randomUUID()),
      source: { kind: 'model' as const, provider: 'claude-subscription', model: 'claude-sonnet-5' },
      content: [{ type: 'reasoning' as const, text: 'pondering' }],
    }
    const user = createUserMessage({ content: [{ type: 'text', text: 'hi' }], source: { kind: 'user' as const, user: 'U' } })
    const wire = serializeMessages([user, reasoningOnly], false)
    expect(wire.some(m => m.role === 'assistant')).toBe(false)
  })

  test('orphaned tool_use (pruned result) gets a placeholder tool_result before the next turn', () => {
    const orphan = {
      role: 'assistant' as const,
      id: MessageId(randomUUID()),
      source: { kind: 'model' as const, provider: 'claude-subscription', model: 'claude-sonnet-5' },
      content: [
        { type: 'tool-call' as const, id: ToolCallId('call_orphan'), name: 'bash', arguments: '{}' },
      ],
    }
    const nextUser = createUserMessage({ content: [{ type: 'text', text: 'continue' }], source: { kind: 'user' as const, user: 'U' } })
    const wire = serializeMessages([orphan, nextUser], false)
    const users = wire.filter(m => m.role === 'user')
    const withPlaceholder = users.find(m => Array.isArray(m.content)
      && ((m.content as unknown) as Array<Record<string, unknown>>).some(b => b.type === 'tool_result' && b.tool_use_id === 'call_orphan' && b.is_error === true))
    expect(withPlaceholder).toBeDefined()
    const withText = users.find(m => Array.isArray(m.content)
      && ((m.content as unknown) as Array<Record<string, unknown>>).some(b => b.type === 'text' && b.text === 'continue'))
    expect(withText).toBeDefined()
  })

  test('multi-tool requests prefix tool names', () => {
    const assistant = {
      role: 'assistant' as const,
      id: MessageId(randomUUID()),
      source: { kind: 'model' as const, provider: 'claude-subscription', model: 'claude-sonnet-5' },
      content: [
        { type: 'tool-call' as const, id: ToolCallId('call_11'), name: 'bash', arguments: '{}' },
      ],
    }
    const wire = serializeMessages([assistant], true)
    expect((wire[0].content as Array<{ name: string }>)[0].name).toBe('mcp_bash')
  })
})

describe('serializeRequest', () => {
  test('system is identity-led; other system text relocates to first user message', async () => {
    const system = createSystemMessage('You are a helpful assistant.', 'test')
    const user = createUserMessage({ content: [{ type: 'text', text: 'do the thing' }], source: { kind: 'user' as const, user: 'U' } })
    const body = await serializeRequest({
      provider: 'claude-subscription',
      model: 'claude-sonnet-5',
      messages: [system, user],
      system: 'extra harness system text',
      maxTokens: 1000,
    })
    expect(body.system).toBeInstanceOf(Array)
    const systemBlocks = body.system as Array<{ type: string; text: string }>
    expect(systemBlocks[1].text).toBe(SYSTEM_IDENTITY)
    expect(systemBlocks[0].text.startsWith('x-anthropic-billing-header')).toBe(true)
    expect(systemBlocks.filter(b => b.text === 'extra harness system text')).toHaveLength(0)
    const firstUser = body.messages.find(m => m.role === 'user')
    expect(typeof firstUser?.content === 'string' ? firstUser.content : '').toContain('extra harness system text')
  })

  test('identity-only system stays in system[]', async () => {
    const user = createUserMessage({ content: [{ type: 'text', text: 'hi' }], source: { kind: 'user' as const, user: 'U' } })
    const body = await serializeRequest({
      provider: 'claude-subscription',
      model: 'claude-sonnet-5',
      messages: [user],
      system: SYSTEM_IDENTITY,
    })
    const systemBlocks = body.system as Array<{ text: string }>
    expect(systemBlocks.filter(b => b.text === SYSTEM_IDENTITY)).toHaveLength(1)
  })

  test('effort maps to output_config for supporting models only', async () => {
    const user = createUserMessage({ content: [{ type: 'text', text: 'hi' }], source: { kind: 'user' as const, user: 'U' } })
    const base = {
      provider: 'claude-subscription',
      model: 'claude-sonnet-5',
      messages: [user],
      reasoningEffort: 'high' as const,
    }
    const sonnet = await serializeRequest(base as never)
    expect((sonnet as any).output_config).toEqual({ effort: 'high' })

    const haiku = await serializeRequest({ ...base, model: 'claude-haiku-4-5' } as never)
    expect((haiku as any).output_config).toBeUndefined()

    const opus48 = await serializeRequest({ ...base, model: 'claude-opus-4-8' } as never)
    expect((opus48 as any).output_config).toEqual({ effort: 'high' })
  })

  test('branded effort id serializes to the wire effort value', async () => {
    const user = createUserMessage({ content: [{ type: 'text', text: 'hi' }], source: { kind: 'user' as const, user: 'U' } })
    const { ReasoningEffortId } = require('@deepseek-ai/dsh-llm')
    const body = await serializeRequest({
      provider: 'claude-subscription',
      model: 'claude-sonnet-5',
      messages: [user],
      reasoningEffort: ReasoningEffortId('low'),
    } as never)
    expect((body as any).output_config).toEqual({ effort: 'low' })
  })

  test('off effort never reaches the wire', async () => {
    const user = createUserMessage({ content: [{ type: 'text', text: 'hi' }], source: { kind: 'user' as const, user: 'U' } })
    const body = await serializeRequest({
      provider: 'claude-subscription',
      model: 'claude-sonnet-5',
      messages: [user],
      reasoningEffort: 'off',
    } as never)
    expect((body as any).output_config).toBeUndefined()
  })

  test('tools map to input_schema with mcp_ prefix when multiple', async () => {
    const user = createUserMessage({ content: [{ type: 'text', text: 'hi' }], source: { kind: 'user' as const, user: 'U' } })
    const body = await serializeRequest({
      provider: 'claude-subscription',
      model: 'claude-sonnet-5',
      messages: [user],
      tools: [
        { name: 'Bash', description: 'run', parameters: { type: 'object' } },
        { name: 'Read', description: 'read', parameters: { type: 'object' } },
      ],
    })
    expect(body.tools).toEqual([
      { name: 'mcp_Bash', description: 'run', input_schema: { type: 'object' } },
      { name: 'mcp_Read', description: 'read', input_schema: { type: 'object' } },
    ])
  })

  test('defaults: stream true, max_tokens fallback', async () => {
    const user = createUserMessage({ content: [{ type: 'text', text: 'hi' }], source: { kind: 'user' as const, user: 'U' } })
    const body = await serializeRequest({
      provider: 'claude-subscription',
      model: 'claude-sonnet-5',
      messages: [user],
    })
    expect(body.stream).toBe(true)
    expect(body.max_tokens).toBe(32_000)
  })
})

  test('billing header is injected as system[0] with cch and version suffix', async () => {
    const user = createUserMessage({ content: [{ type: 'text', text: 'hello world' }], source: { kind: 'user' as const, user: 'U' } })
    const body = await serializeRequest({
      provider: 'claude-subscription',
      model: 'claude-sonnet-5',
      messages: [user],
    } as never)
    const system = (body as any).system as Array<{ type: string; text: string }>
    expect(system[0].text.startsWith('x-anthropic-billing-header: cc_version=')).toBe(true)
    expect(system[0].text).toContain('cc_entrypoint=sdk-cli')
    expect(system[0].text).toMatch(/cch=[0-9a-f]{5}/)
    expect(system[1].text).toBe("You are Claude Code, Anthropic's official CLI for Claude.")
  })

  test('billing header cch matches sha256 of first user text', async () => {
    const { createHash } = require('node:crypto')
    const text = 'test message for hashing'
    const user = createUserMessage({ content: [{ type: 'text', text }], source: { kind: 'user' as const, user: 'U' } })
    const body = await serializeRequest({
      provider: 'claude-subscription',
      model: 'claude-sonnet-5',
      messages: [user],
    } as never)
    const system = (body as any).system as Array<{ type: string; text: string }>
    const expected = createHash('sha256').update(text).digest('hex').slice(0, 5)
    expect(system[0].text).toContain(`cch=${expected}`)
  })

describe('empty content sanitization', () => {
  test('assistant whitespace-only text is dropped from the wire', async () => {
    const assistant = {
      role: 'assistant' as const,
      id: MessageId(randomUUID()),
      source: { kind: 'model' as const, provider: 'claude-subscription', model: 'claude-sonnet-5' },
      content: [{ type: 'text' as const, text: '   ' }],
    }
    const user = createUserMessage({ content: [{ type: 'text', text: 'hi' }], source: { kind: 'user' as const, user: 'U' } })
    const wire = serializeMessages([user, assistant], false)
    expect(wire.some(m => m.role === 'assistant')).toBe(false)
  })

  test('assistant empty text with tool call drops the text block entirely', async () => {
    const assistant = {
      role: 'assistant' as const,
      id: MessageId(randomUUID()),
      source: { kind: 'model' as const, provider: 'claude-subscription', model: 'claude-sonnet-5' },
      content: [
        { type: 'text' as const, text: '' },
        { type: 'tool-call' as const, id: ToolCallId('c1'), name: 'bash', arguments: '{}' },
      ],
    }
    const wire = serializeMessages([assistant], false)
    expect(wire[0].content).toEqual([{ type: 'tool_use', id: 'c1', name: 'bash', input: {} }])
  })

  test('user empty text message is dropped from the wire', async () => {
    const user = {
      role: 'user' as const,
      id: MessageId(randomUUID()),
      source: { kind: 'user' as const, user: 'U' },
      content: [{ type: 'text' as const, text: '' }],
    }
    const realUser = createUserMessage({ content: [{ type: 'text', text: 'hi' }], source: { kind: 'user' as const, user: 'U' } })
    const wire = serializeMessages([realUser, user], false)
    expect(wire).toHaveLength(1)
  })
})

describe('boundary sanitizer', () => {
  test('empty assistant turn (interrupted stream) is dropped, not sent as empty text', async () => {
    const user = createUserMessage({ content: [{ type: 'text', text: 'hola' }], source: { kind: 'user' as const, user: 'U' } })
    const emptyAssistant = {
      role: 'assistant' as const,
      id: MessageId(randomUUID()),
      source: { kind: 'model' as const, provider: 'claude-subscription', model: 'claude-sonnet-5' },
      content: [{ type: 'text' as const, text: '' }],
    }
    const next = createUserMessage({ content: [{ type: 'text', text: 'continua' }], source: { kind: 'user' as const, user: 'U' } })
    const wire = serializeMessages([user, emptyAssistant, next], false)
    expect(wire.some(m => m.role === 'assistant')).toBe(false)
    expect(wire.map(m => m.role)).toEqual(['user'])
    expect(typeof wire[0].content === 'string' ? wire[0].content : '').toContain('hola')
  })

  test('consecutive user messages merge into one (API role alternation)', async () => {
    const u1 = createUserMessage({ content: [{ type: 'text', text: 'primero' }], source: { kind: 'user' as const, user: 'U' } })
    const u2 = createUserMessage({ content: [{ type: 'text', text: 'segundo' }], source: { kind: 'user' as const, user: 'U' } })
    const wire = serializeMessages([u1, u2], false)
    expect(wire).toHaveLength(1)
    expect(wire[0].content).toContain('primero')
    expect(wire[0].content).toContain('segundo')
  })

  test('consecutive assistant text turns are both kept (API merges same-role turns)', async () => {
    const mk = (id: string, text: string) => ({
      role: 'assistant' as const,
      id: MessageId(id),
      source: { kind: 'model' as const, provider: 'claude-subscription', model: 'claude-sonnet-5' },
      content: [{ type: 'text' as const, text }],
    })
    const user = createUserMessage({ content: [{ type: 'text', text: 'hola' }], source: { kind: 'user' as const, user: 'U' } })
    const wire = serializeMessages([user, mk(randomUUID(), 'parte uno'), mk(randomUUID(), 'parte dos')], false)
    const asst = wire.filter(m => m.role === 'assistant')
    expect(asst).toHaveLength(2)
    expect(asst[0].content).toEqual([{ type: 'text', text: 'parte uno' }])
    expect(asst[1].content).toEqual([{ type: 'text', text: 'parte dos' }])
  })

  test('orphaned tool_use at end of history gets a placeholder result', async () => {
    const user = createUserMessage({ content: [{ type: 'text', text: 'run ls' }], source: { kind: 'user' as const, user: 'U' } })
    const assistant = {
      role: 'assistant' as const,
      id: MessageId(randomUUID()),
      source: { kind: 'model' as const, provider: 'claude-subscription', model: 'claude-sonnet-5' },
      content: [{ type: 'tool-call' as const, id: ToolCallId('orphan1'), name: 'bash', arguments: '{"command":"ls"}' }],
    }
    const wire = serializeMessages([user, assistant], false)
    const placeholder = wire.find(m => m.role === 'user' && Array.isArray(m.content)
      && ((m.content as unknown) as Array<Record<string, unknown>>).some(b => b.type === 'tool_result' && b.tool_use_id === 'orphan1' && b.is_error === true))
    expect(placeholder).toBeDefined()
  })

  test('tool result in a later user message (after a merged user) is hoisted to the immediate next user', async () => {
    // Reproduces the real failing session shape:
    //   user (file)
    //   user (model-notice)        <- harness merges these two into one user message
    //   assistant (tool_use A, tool_use B)
    //   user (result A)            <- separate user messages for each result
    //   user (result B)
    // After merge, the wire would have:
    //   user (merged)
    //   assistant (A, B)
    //   user (result A)            <- only A here, B is in the next user message
    //   user (result B)
    // Anthropic requires BOTH A and B in the immediate next user message.
    const asrc = { kind: 'model' as const, provider: 'claude-subscription', model: 'claude-sonnet-5' }
    const mergedUser = createUserMessage({
      content: [{ type: 'text' as const, text: 'file + model notice (merged)' }],
      source: { kind: 'user' as const, user: 'U' },
    })
    const assistant = {
      role: 'assistant' as const, id: MessageId(randomUUID()), source: asrc,
      content: [
        { type: 'tool-call' as const, id: ToolCallId('idA'), name: 'bash', arguments: '{}' },
        { type: 'tool-call' as const, id: ToolCallId('idB'), name: 'bash', arguments: '{}' },
      ],
    }
    const resultA = createUserMessage({
      content: [{ type: 'tool-result' as const, toolCallId: ToolCallId('idA'), content: [{ type: 'text' as const, text: 'rA' }] }],
      source: { kind: 'user' as const, user: 'U' },
    })
    const resultB = createUserMessage({
      content: [{ type: 'tool-result' as const, toolCallId: ToolCallId('idB'), content: [{ type: 'text' as const, text: 'rB' }] }],
      source: { kind: 'user' as const, user: 'U' },
    })
    const wire = serializeMessages([mergedUser, assistant, resultA, resultB], true)
    // The assistant is at index 1; the immediate next user (index 2) must contain BOTH results.
    const asstIdx = wire.findIndex(m => m.role === 'assistant')
    expect(asstIdx).toBe(1)
    const nextUser = wire[asstIdx + 1]!
    expect(nextUser.role).toBe('user')
    const results = (nextUser.content as unknown as Array<Record<string, unknown>>).filter(b => b.type === 'tool_result').map(b => b.tool_use_id)
    expect(results).toContain('idA')
    expect(results).toContain('idB')
  })

  test('leading system messages are stripped from the wire messages', async () => {
    const system = {
      role: 'system' as const,
      id: MessageId(randomUUID()),
      source: { kind: 'plugin' as const, plugin: 'x', section: 's' },
      content: [{ type: 'text' as const, text: 'sys text' }],
    }
    const user = createUserMessage({ content: [{ type: 'text', text: 'hola' }], source: { kind: 'user' as const, user: 'U' } })
    const wire = serializeMessages([system, user], false)
    expect(wire.some(m => m.role === 'system')).toBe(false)
    expect(wire[0].role).toBe('user')
  })
})

describe('image serialization', () => {
  // A 1x1 transparent PNG, base64.
  const PNG_B64 = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=='
  const pngBytes = Buffer.from(PNG_B64, 'base64')

  function fakeAttachments(bytes = pngBytes, mediaType = 'image/png') {
    return {
      readImageRequest: async (_ref: any, _policy: any, _signal?: any) => ({ data: bytes, mediaType }),
    }
  }

  function imageRef(id = 'att_1'): any {
    return { attachmentId: id, mediaType: 'image/png', bytes: pngBytes.length, width: 1, height: 1 }
  }

  test('user image becomes an Anthropic base64 image block', async () => {
    const user = createUserMessage({
      content: [
        { type: 'image', attachment: imageRef('att_img') },
        { type: 'text', text: 'what is this?' },
      ],
      source: { kind: 'user' as const, user: 'U' },
    })
    const body = await serializeRequest({
      provider: 'claude-subscription', model: 'claude-sonnet-5', messages: [user], maxTokens: 1000,
    }, fakeAttachments())
    const firstUser = body.messages.find(m => m.role === 'user')
    const content = firstUser!.content as Array<Record<string, any>>
    const img = content.find(b => b.type === 'image')
    expect(img).toBeDefined()
    expect(img!.source).toEqual({ type: 'base64', media_type: 'image/png', data: PNG_B64 })
    expect(content.find(b => b.type === 'text')?.text).toBe('what is this?')
  })

  test('tool-result image is emitted inside the tool_result content array', async () => {
    const user = createUserMessage({
      content: [
        { type: 'tool-result', toolCallId: ToolCallId('call_img'), content: [{ type: 'image', attachment: imageRef('att_tr') }] },
      ],
      source: { kind: 'user' as const, user: 'U' },
    })
    const body = await serializeRequest({
      provider: 'claude-subscription', model: 'claude-sonnet-5', messages: [user], maxTokens: 1000,
    }, fakeAttachments())
    const firstUser = body.messages.find(m => m.role === 'user')
    const blocks = firstUser!.content as Array<Record<string, any>>
    const tr = blocks.find(b => b.type === 'tool_result')
    expect(tr).toBeDefined()
    expect(Array.isArray(tr!.content)).toBe(true)
    expect(tr!.content.some((b: any) => b.type === 'image' && b.source.type === 'base64')).toBe(true)
  })

  test('duplicate image refs across the request are resolved once', async () => {
    let reads = 0
    const attachments = {
      readImageRequest: async (_ref: any) => { reads++; return { data: pngBytes, mediaType: 'image/png' } },
    }
    const user1 = createUserMessage({ content: [{ type: 'image', attachment: imageRef('att_dup') }], source: { kind: 'user' as const, user: 'U' } })
    const user2 = createUserMessage({ content: [{ type: 'image', attachment: imageRef('att_dup') }], source: { kind: 'user' as const, user: 'U' } })
    await serializeRequest({
      provider: 'claude-subscription', model: 'claude-sonnet-5', messages: [user1, user2], maxTokens: 1000,
    }, attachments)
    expect(reads).toBe(1)
  })

  test('image present but no attachment service throws UNSUPPORTED_CONTENT', async () => {
    const user = createUserMessage({ content: [{ type: 'image', attachment: imageRef('att_x') }], source: { kind: 'user' as const, user: 'U' } })
    await expect(serializeRequest({
      provider: 'claude-subscription', model: 'claude-sonnet-5', messages: [user], maxTokens: 1000,
    })).rejects.toThrow(/no attachment service/)
  })

  test('text-only request performs no attachment reads', async () => {
    let reads = 0
    const attachments = { readImageRequest: async () => { reads++; return { data: pngBytes, mediaType: 'image/png' } } }
    const user = createUserMessage({ content: [{ type: 'text', text: 'hi' }], source: { kind: 'user' as const, user: 'U' } })
    await serializeRequest({
      provider: 'claude-subscription', model: 'claude-sonnet-5', messages: [user], maxTokens: 1000,
    }, attachments)
    expect(reads).toBe(0)
  })
})

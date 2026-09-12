import { describe, expect, test } from 'bun:test'
import { AnthropicEventTranslator } from '../src/translate.ts'
import { createSseParser } from '../src/sse.ts'
import type { WireSseEvent } from '../src/types.ts'

function eventsFrom(sse: string): WireSseEvent[] {
  const parser = createSseParser()
  const encoder = new TextEncoder()
  const out: WireSseEvent[] = [...parser.push(encoder.encode(sse))]
  return out.concat(parser.end())
}

describe('SSE parser', () => {
  test('parses a basic event stream', () => {
    const events = eventsFrom(
      'event: message_start\ndata: {"type":"message_start","message":{"id":"m1","usage":{"input_tokens":10}}}\n\n'
      + 'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}\n\n'
      + 'event: message_stop\ndata: {"type":"message_stop"}\n\n',
    )
    expect(events.map(e => e.type)).toEqual(['message_start', 'content_block_delta', 'message_stop'])
  })

  test('handles events split across chunks', () => {
    const parser = createSseParser()
    const encoder = new TextEncoder()
    const full = 'data: {"type":"message_stop"}\n\n'
    const half = Math.ceil(full.length / 2)
    const first = parser.push(encoder.encode(full.slice(0, half)))
    const second = parser.push(encoder.encode(full.slice(half)))
    const third = parser.end()
    expect([...first, ...second, ...third].map(e => e.type)).toEqual(['message_stop'])
  })

  test('ignores non-JSON frames and comments', () => {
    const events = eventsFrom(
      ': keep-alive\n\n'
      + 'data: not-json\n\n'
      + 'data: {"type":"ping"}\n\n',
    )
    expect(events.map(e => e.type)).toEqual(['ping'])
  })
})

describe('AnthropicEventTranslator', () => {
  function run(sse: string, knownTools?: ReadonlySet<string>) {
    const translator = new AnthropicEventTranslator(knownTools)
    const chunks: import("@deepseek-ai/dsh-llm").StreamChunk[] = []
    for (const event of eventsFrom(sse)) chunks.push(...translator.translate(event))
    return chunks
  }

  test('block-end carries the accumulated text (harness builds the message from block-end, not deltas)', () => {
    const chunks = run(
      'data: {"type":"message_start","message":{"id":"m6","usage":{"input_tokens":1}}}\n\n'
      + 'data: {"type":"content_block_start","index":0,"content_block":{"type":"text"}}\n\n'
      + 'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hola"}}\n\n'
      + 'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" mundo"}}\n\n'
      + 'data: {"type":"content_block_stop","index":0}\n\n'
      + 'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5}}\n\n'
      + 'data: {"type":"message_stop"}\n\n',
    )
    const end = chunks.find((c): c is Extract<import('@deepseek-ai/dsh-llm').StreamChunk, { type: 'block-end' }> => c.type === 'block-end')
    expect(end).toBeDefined()
    expect(end?.block).toEqual({ type: 'text', text: 'hola mundo' })
  })

  test('thinking-then-text: single harness text block carries thinking + text', () => {
    const chunks = run(
      'data: {"type":"message_start","message":{"id":"m7","usage":{"input_tokens":1}}}\n\n'
      + 'data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking"}}\n\n'
      + 'data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"pense"}}\n\n'
      + 'data: {"type":"content_block_stop","index":0}\n\n'
      + 'data: {"type":"content_block_start","index":1,"content_block":{"type":"text"}}\n\n'
      + 'data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"respuesta"}}\n\n'
      + 'data: {"type":"content_block_stop","index":1}\n\n'
      + 'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":10}}\n\n'
      + 'data: {"type":"message_stop"}\n\n',
    )
    const starts = chunks.filter((c): c is Extract<import('@deepseek-ai/dsh-llm').StreamChunk, { type: 'block-start' }> => c.type === 'block-start')
    expect(starts).toHaveLength(1)
    const ends = chunks.filter((c): c is Extract<import('@deepseek-ai/dsh-llm').StreamChunk, { type: 'block-end' }> => c.type === 'block-end')
    expect(ends).toHaveLength(1)
    expect(ends[0]?.block).toEqual({ type: 'text', text: 'penserespuesta' })
  })

  test('text-only message yields block-scoped chunks, usage and stop finish', () => {
    const chunks = run(
      'data: {"type":"message_start","message":{"id":"m1","usage":{"input_tokens":10,"cache_read_input_tokens":2}}}\n\n'
      + 'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n\n'
      + 'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hel"}}\n\n'
      + 'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"lo"}}\n\n'
      + 'data: {"type":"content_block_stop","index":0}\n\n'
      + 'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5}}\n\n'
      + 'data: {"type":"message_stop"}\n\n',
    )
    expect(chunks).toEqual([
      { type: 'block-start', index: 0, blockType: 'text' },
      { type: 'text-delta', index: 0, text: 'hel' },
      { type: 'text-delta', index: 0, text: 'lo' },
      { type: 'block-end', index: 0, block: { type: 'text', text: 'hello' } },
      { type: 'usage', usage: { inputTokens: 8, outputTokens: 5, cacheReadTokens: 2 } },
      { type: 'finish', reason: { kind: 'stop' } },
    ])
  })

  test('tool call streams accumulate arguments and emit a tool-call block', () => {
    const chunks = run(
      'data: {"type":"message_start","message":{"id":"m2","usage":{"input_tokens":1}}}\n\n'
      + 'data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"call_1","name":"bash"}}\n\n'
      + 'data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\\"cmd\\""}}\n\n'
      + 'data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":": \\"ls\\"}"}}\n\n'
      + 'data: {"type":"content_block_stop","index":0}\n\n'
      + 'data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":9}}\n\n'
      + 'data: {"type":"message_stop"}\n\n',
    )
    expect(chunks[0]).toMatchObject({ type: 'tool-call-delta', index: 0, id: 'call_1', name: 'bash' })
    const toolEnd = chunks.find(c => (c as { type: string }).type === 'block-end') as {
      block: { type: string; name: string; arguments: string }
    }
    expect(toolEnd.block.type).toBe('tool-call')
    expect(toolEnd.block.name).toBe('bash')
    expect(toolEnd.block.arguments).toBe('{"cmd": "ls"}')
    expect(chunks[chunks.length - 1]).toEqual({ type: 'finish', reason: { kind: 'tool-calls' } })
  })

  test('thinking deltas are carried and delivered with the text block', () => {
    const chunks = run(
      'data: {"type":"message_start","message":{"id":"m3","usage":{"input_tokens":1}}}\n\n'
      + 'data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking"}}\n\n'
      + 'data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"ponder"}}\n\n'
      + 'data: {"type":"content_block_stop","index":0}\n\n'
      + 'data: {"type":"content_block_start","index":1,"content_block":{"type":"text"}}\n\n'
      + 'data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"hola"}}\n\n'
      + 'data: {"type":"content_block_stop","index":1}\n\n'
      + 'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":3}}\n\n'
      + 'data: {"type":"message_stop"}\n\n',
    )
    expect(chunks[0]).toEqual({ type: 'block-start', index: 0, blockType: 'text' })
    expect(chunks[1]).toEqual({ type: 'text-delta', index: 0, text: 'ponder' })
    expect(chunks[2]).toEqual({ type: 'text-delta', index: 0, text: 'hola' })
    const end = chunks.find((c): c is Extract<import('@deepseek-ai/dsh-llm').StreamChunk, { type: 'block-end' }> => c.type === 'block-end')
    expect(end?.block).toEqual({ type: 'text', text: 'ponderhola' })
  })

  test('thinking-only turn: thinking carried into a text block, finish stop (no empty assistant)', () => {
    const chunks = run(
      'data: {"type":"message_start","message":{"id":"m5","usage":{"input_tokens":1}}}\n\n'
      + 'data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking"}}\n\n'
      + 'data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"deep thought"}}\n\n'
      + 'data: {"type":"signature_delta","index":0,"delta":{"signature":"abc"}}\n\n'
      + 'data: {"type":"content_block_stop","index":0}\n\n'
      + 'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":10}}\n\n'
      + 'data: {"type":"message_stop"}\n\n',
    )
    // thinking is carried and delivered as a text block (non-empty), so the
    // harness stores a non-empty assistant — no empty-text turn to 400 on replay.
    const text = chunks.filter((c): c is Extract<import('@deepseek-ai/dsh-llm').StreamChunk, { type: 'text-delta' }> => c.type === 'text-delta').map(c => c.text).join('')
    expect(text).toBe('deep thought')
    const last = chunks[chunks.length - 1]
    expect(last).toEqual({ type: 'finish', reason: { kind: 'stop' } })
  })

  test('in-band error on message_delta finishes with an error', () => {
    const chunks = run(
      'data: {"type":"message_start","message":{"id":"m4","usage":{"input_tokens":1}}}\n\n'
      + 'data: {"type":"message_delta","delta":{"error":{"type":"overloaded_error","message":"Overloaded"}}}\n\n',
    )
    expect(chunks[chunks.length - 1]).toEqual({
      type: 'finish',
      reason: { kind: 'error', failure: { message: 'Overloaded', code: 'overloaded_error' } },
    })
  })

  test('stream ending without message_start is an EMPTY_RESPONSE finish', () => {
    const chunks = run('data: {"type":"message_stop"}\n\n')
    expect(chunks).toEqual([{
      type: 'finish',
      reason: { kind: 'error', failure: { message: 'Claude returned no message', code: 'EMPTY_RESPONSE' } },
    }])
  })

  test('max_tokens stop reason maps to max-tokens finish', () => {
    const chunks = run(
      'data: {"type":"message_start","message":{"id":"m5","usage":{"input_tokens":1}}}\n\n'
      + 'data: {"type":"message_delta","delta":{"stop_reason":"max_tokens"},"usage":{"output_tokens":10}}\n\n'
      + 'data: {"type":"message_stop"}\n\n',
    )
    expect(chunks[chunks.length - 1]).toEqual({ type: 'finish', reason: { kind: 'max-tokens' } })
  })

  test('hallucinated mcp_ prefix on a known tool is remapped to the registered name', () => {
    const known = new Set(['bash', 'glob', 'mcp__gitlab__search'])
    const sse =
      'data: {"type":"message_start","message":{"id":"m8","usage":{"input_tokens":1}}}\n\n'
      + 'data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_abc","name":"mcp_bash"}}\n\n'
      + 'data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\\"command\\":\\"ls\\"}"}}\n\n'
      + 'data: {"type":"content_block_stop","index":0}\n\n'
      + 'data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":5}}\n\n'
      + 'data: {"type":"message_stop"}\n\n'
    const chunks = run(sse, known)
    const first = chunks[0]
    // tool_use start emits a tool-call-delta carrying the (remapped) name
    expect(first).toEqual({ type: 'tool-call-delta', index: 0, id: expect.anything(), name: 'bash', argumentsDelta: '' })
  })

  test('legit mcp__server__tool name is left untouched', () => {
    const known = new Set(['bash', 'mcp__gitlab__search'])
    const sse =
      'data: {"type":"message_start","message":{"id":"m9","usage":{"input_tokens":1}}}\n\n'
      + 'data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_abc","name":"mcp__gitlab__search"}}\n\n'
      + 'data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\\"q\\":\\"x\\"}"}}\n\n'
      + 'data: {"type":"content_block_stop","index":0}\n\n'
      + 'data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":5}}\n\n'
      + 'data: {"type":"message_stop"}\n\n'
    const chunks = run(sse, known)
    expect(chunks[0]).toEqual({ type: 'tool-call-delta', index: 0, id: expect.anything(), name: 'mcp__gitlab__search', argumentsDelta: '' })
  })

  test('unknown tool with no known stripped form is passed through as-is', () => {
    const known = new Set(['bash'])
    const sse =
      'data: {"type":"message_start","message":{"id":"m10","usage":{"input_tokens":1}}}\n\n'
      + 'data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_abc","name":"mcp_frobnicate"}}\n\n'
      + 'data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\\"x\\":1}"}}\n\n'
      + 'data: {"type":"content_block_stop","index":0}\n\n'
      + 'data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":5}}\n\n'
      + 'data: {"type":"message_stop"}\n\n'
    const chunks = run(sse, known)
    expect(chunks[0]).toEqual({ type: 'tool-call-delta', index: 0, id: expect.anything(), name: 'mcp_frobnicate', argumentsDelta: '' })
  })
})

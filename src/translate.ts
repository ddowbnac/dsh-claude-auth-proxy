import type {
  ContentBlock,
  FinishReason,
  LlmFailure,
  StreamChunk,
  TokenUsage,
} from '@deepseek-ai/dsh-llm'
import { ToolCallId as ToolCallIdBrand } from '@deepseek-ai/dsh-llm'
import type { WireSseEvent, WireUsage } from './types.ts'

export class AnthropicEventTranslator {
  constructor(private readonly knownTools: ReadonlySet<string> = new Set()) {}

  private nextBlockIndex = 0
  private openTextIndex: number | undefined
  private openText = ''
  private thinkingCarry = ''
  private emittedBlocks = 0
  private openToolCalls = new Map<number, { blockIndex: number; id: string; name: string; args: string }>()
  private inputTokens = 0
  private cacheReadTokens = 0
  private cacheWriteTokens = 0
  private outputTokens = 0
  private stopReason: string | undefined
  private sawMessageStart = false
  private finished = false

  /**
   * The model sometimes hallucinates an `mcp_` namespace prefix onto
   * built-in tools (it sees many `mcp__server__tool` names in the list and
   * generalizes the prefix onto `bash`, `glob`, etc.). A call to `mcp_bash`
   * then fails with `unknown tool`. Remap the hallucinated name back to the
   * registered one when stripping the prefix yields a known tool.
   */
  private resolveToolName(name: string): string {
    if (name.length === 0 || this.knownTools.has(name)) return name
    // `mcp_bash` -> `bash`
    if (name.startsWith('mcp_')) {
      const stripped = name.slice('mcp_'.length)
      if (this.knownTools.has(stripped)) return stripped
    }
    return name
  }

  translate(event: WireSseEvent): StreamChunk[] {
    if (this.finished) return []
    switch (event.type) {
      case 'message_start':
        return this.onMessageStart(event)
      case 'content_block_start':
        return this.onContentBlockStart(event)
      case 'content_block_delta':
        return this.onContentBlockDelta(event)
      case 'content_block_stop':
        return this.onContentBlockStop(event)
      case 'message_delta':
        return this.onMessageDelta(event)
      case 'message_stop':
        return this.onMessageStop()
      case 'error':
        return this.onStreamError(event)
      case 'ping':
      default:
        return []
    }
  }

  private onMessageStart(event: WireSseEvent): StreamChunk[] {
    this.sawMessageStart = true
    const message = (event.message ?? {}) as Record<string, unknown>
    const usage = (message.usage ?? {}) as WireUsage
    this.inputTokens = usage.input_tokens ?? 0
    this.cacheReadTokens = usage.cache_read_input_tokens ?? 0
    this.cacheWriteTokens = usage.cache_creation_input_tokens ?? 0
    return []
  }

  private onContentBlockStart(event: WireSseEvent): StreamChunk[] {
    const index = event.index as number
    const block = (event.content_block ?? {}) as Record<string, unknown>
    switch (block.type) {
      case 'text': {
        this.openTextIndex = this.nextBlockIndex++
        this.openText = ''
        const carry = this.thinkingCarry
        this.thinkingCarry = ''
        const idx = this.openTextIndex
        const chunks: StreamChunk[] = [{ type: 'block-start', index: idx, blockType: 'text' }]
        if (carry.length > 0) {
          this.openText = carry
          chunks.push({ type: 'text-delta', index: idx, text: carry })
        }
        return chunks
      }
      case 'thinking':
      case 'redacted_thinking':
        return []
      case 'tool_use': {
        const id = typeof block.id === 'string' ? block.id : `toolu_${index}`
        const rawName = typeof block.name === 'string' ? block.name : ''
        const name = this.resolveToolName(rawName)
        const blockIndex = this.nextBlockIndex++
        this.openToolCalls.set(index, { blockIndex, id, name, args: '' })
        return [{
          type: 'tool-call-delta',
          index: blockIndex,
          id: ToolCallIdBrand(id),
          name,
          argumentsDelta: '',
        }]
      }
      default:
        return []
    }
  }

  private onContentBlockDelta(event: WireSseEvent): StreamChunk[] {
    const index = event.index as number
    const delta = (event.delta ?? {}) as Record<string, unknown>
    switch (delta.type) {
      case 'text_delta': {
        if (this.openTextIndex === undefined) return []
        const text = typeof delta.text === 'string' ? delta.text : ''
        if (text.length === 0) return []
        this.openText += text
        return [{ type: 'text-delta', index: this.openTextIndex, text }]
      }
      case 'thinking_delta': {
        const text = typeof delta.thinking === 'string' ? delta.thinking : ''
        if (text.length === 0) return []
        this.thinkingCarry += text
        return []
      }
      case 'input_json_delta': {
        const open = this.openToolCalls.get(index)
        if (open === undefined) return []
        const partial = typeof delta.partial_json === 'string' ? delta.partial_json : ''
        if (partial.length === 0) return []
        open.args += partial
        return [{ type: 'tool-call-delta', index: open.blockIndex, id: ToolCallIdBrand(open.id), argumentsDelta: partial }]
      }
      case 'signature_delta':
        return []
      default:
        return []
    }
  }

  private onContentBlockStop(event: WireSseEvent): StreamChunk[] {
    const index = event.index as number
    const chunks: StreamChunk[] = []
    const tool = this.openToolCalls.get(index)
    if (tool !== undefined) {
      const block: ContentBlock = {
        type: 'tool-call',
        id: ToolCallIdBrand(tool.id),
        name: tool.name,
        arguments: tool.args.length > 0 ? tool.args : '{}',
      }
      this.emittedBlocks++
      chunks.push({ type: 'block-end', index: tool.blockIndex, block })
      this.openToolCalls.delete(index)
      return chunks
    }
    if (this.openTextIndex !== undefined) {
      this.emittedBlocks++
      chunks.push({ type: 'block-end', index: this.openTextIndex, block: { type: 'text', text: this.openText } })
      this.openTextIndex = undefined
      this.openText = ''
      return chunks
    }
    return []
  }

  private onMessageDelta(event: WireSseEvent): StreamChunk[] {
    const delta = (event.delta ?? {}) as Record<string, unknown>
    if (typeof delta.stop_reason === 'string') this.stopReason = delta.stop_reason
    const usage = (event.usage ?? {}) as WireUsage
    if (typeof usage.output_tokens === 'number') this.outputTokens = usage.output_tokens
    if (typeof usage.input_tokens === 'number') this.inputTokens = usage.input_tokens
    if (typeof usage.cache_read_input_tokens === 'number') this.cacheReadTokens = usage.cache_read_input_tokens
    if (typeof usage.cache_creation_input_tokens === 'number') this.cacheWriteTokens = usage.cache_creation_input_tokens
    if (typeof delta.error === 'object' && delta.error !== null) {
      const error = delta.error as Record<string, unknown>
      const failure: LlmFailure = {
        message: typeof error.message === 'string' ? error.message : 'stream error',
        code: typeof error.type === 'string' ? error.type : 'API_ERROR',
      }
      return [this.finish({ kind: 'error', failure })]
    }
    return []
  }

  private onMessageStop(): StreamChunk[] {
    const chunks: StreamChunk[] = []
    if (this.stopReason !== 'max_tokens') {
      for (const [, tool] of this.openToolCalls) {
        this.emittedBlocks++
        chunks.push({
          type: 'block-end',
          index: tool.blockIndex,
          block: {
            type: 'tool-call',
            id: ToolCallIdBrand(tool.id),
            name: tool.name,
            arguments: tool.args.length > 0 ? tool.args : '{}',
          },
        })
      }
    }
    this.openToolCalls.clear()
    if (this.openTextIndex !== undefined) {
      this.emittedBlocks++
      chunks.push({ type: 'block-end', index: this.openTextIndex, block: { type: 'text', text: this.openText } })
      this.openTextIndex = undefined
      this.openText = ''
    } else if (this.thinkingCarry.length > 0) {
      // Thinking-only stream: no text block ever opened. Flush the carried
      // thinking as a non-empty text block so the harness stores a non-empty
      // assistant (an empty-text turn would 400 on replay).
      const blockIndex = this.nextBlockIndex++
      const text = this.thinkingCarry
      this.thinkingCarry = ''
      this.emittedBlocks++
      chunks.push({ type: 'block-start', index: blockIndex, blockType: 'text' })
      chunks.push({ type: 'text-delta', index: blockIndex, text })
      chunks.push({ type: 'block-end', index: blockIndex, block: { type: 'text', text } })
    }
    this.finished = true
    if (!this.sawMessageStart) {
      chunks.push(this.finish({ kind: 'error', failure: { message: 'Claude returned no message', code: 'EMPTY_RESPONSE' } }))
      return chunks
    }
    const usage: TokenUsage = {
      inputTokens: Math.max(0, this.inputTokens - this.cacheReadTokens),
      outputTokens: this.outputTokens,
      ...(this.cacheReadTokens > 0 ? { cacheReadTokens: this.cacheReadTokens } : {}),
      ...(this.cacheWriteTokens > 0 ? { cacheWriteTokens: this.cacheWriteTokens } : {}),
    }
    chunks.push({ type: 'usage', usage })
    const noVisibleBlock = this.emittedBlocks === 0 && this.thinkingCarry.length === 0
    // max_tokens without a visible block is a legitimate truncation (model spent
    // its budget on thinking), not a degenerate empty response.
    if (noVisibleBlock && this.stopReason !== 'max_tokens') {
      chunks.push(this.finish({ kind: 'error', failure: { message: 'Claude returned no content blocks', code: 'EMPTY_RESPONSE' } }))
    } else {
      chunks.push(this.finish(this.mapStopReason(this.stopReason)))
    }
    return chunks
  }

  private onStreamError(event: WireSseEvent): StreamChunk[] {
    const error = (event.error ?? {}) as Record<string, unknown>
    const failure: LlmFailure = {
      message: typeof error.message === 'string' ? error.message : 'stream error',
      code: typeof error.type === 'string' ? error.type : 'API_ERROR',
    }
    this.finished = true
    return [this.finish({ kind: 'error', failure })]
  }

  private mapStopReason(stop: string | undefined): FinishReason {
    switch (stop) {
      case 'tool_use':
        return { kind: 'tool-calls' }
      case 'max_tokens':
        return { kind: 'max-tokens' }
      default:
        return { kind: 'stop' }
    }
  }

  private finish(reason: FinishReason): StreamChunk {
    return { type: 'finish', reason }
  }
}

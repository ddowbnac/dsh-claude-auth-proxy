import { LlmError } from '@deepseek-ai/dsh-llm'
import type { ContentBlock, GenerateOptions, Message, ToolResultBlock } from '@deepseek-ai/dsh-llm'
import { buildBillingHeader } from './billing.ts'
import { SYSTEM_IDENTITY, type WireBlock, type WireImageBlock, type WireMessage, type WireRequest, type WireTool } from './types.ts'
import { supportsEffort } from './model-config.ts'

/**
 * Structural slice of the harness's durable image reference (from
 * `@deepseek-ai/dsh-attachment`). Typed structurally so the plugin does not
 * depend on the attachment package directly — the harness hands us refs shaped
 * exactly like this via `ImageBlock.attachment`.
 */
export interface ImageRef {
  attachmentId: string
  mediaType: string
  bytes: number
  width: number
  height: number
  name?: string
}

/** The attachment service as the harness exposes it: read normalized image bytes. */
export interface AttachmentReader {
  readImageRequest(ref: ImageRef, policy: { maxPixels: number; maxBytes: number }, signal?: AbortSignal): Promise<{ data: Uint8Array; mediaType: string }>
}

/** Resolved image payload, keyed by attachment id, ready for the sync serializer. */
interface ResolvedImage {
  mediaType: string
  dataBase64: string
}

function flattenText(blocks: ContentBlock[]): string {
  return blocks.filter(block => block.type === 'text').map(block => block.text).join('')
}

/** Walk a message's content (including nested tool-result content) for image refs, in order. */
function collectImageRefs(content: readonly ContentBlock[], out: ImageRef[]): void {
  for (const block of content) {
    if (block.type === 'image') out.push(block.attachment)
    else if (block.type === 'tool-result') collectImageRefs(block.content, out)
  }
}

/**
 * Resolve every durable image in the request to base64 bytes in one async pass.
 * Returns a map keyed by attachment id. When no attachments are present (the
 * common text-only case) this is a no-op returning an empty map.
 */
async function resolveImages(
  messages: readonly Message[],
  attachments: AttachmentReader | undefined,
  signal: AbortSignal | undefined,
): Promise<Map<string, ResolvedImage>> {
  const refs: ImageRef[] = []
  for (const message of messages) collectImageRefs(message.content, refs)
  if (refs.length === 0) return new Map()
  if (attachments === undefined) {
    throw new LlmError('The Claude subscription adapter has no attachment service; image input is unavailable.', 'UNSUPPORTED_CONTENT')
  }
  const policy = { maxPixels: 64_000, maxBytes: 5_000_000 }
  const unique = new Map<string, ImageRef>()
  for (const ref of refs) if (!unique.has(ref.attachmentId)) unique.set(ref.attachmentId, ref)
  const entries = [...unique.entries()]
  const projected = await Promise.all(entries.map(([id, ref]) =>
    attachments.readImageRequest(ref, policy, signal).then(data => [id, { mediaType: data.mediaType, dataBase64: Buffer.from(data.data).toString('base64') }] as const)
  ))
  return new Map(projected)
}

const TOOL_NAME_PREFIX = 'mcp_'

function prefixedToolName(name: string, multiTool: boolean): string {
  if (!multiTool) return name
  if (name.startsWith(TOOL_NAME_PREFIX)) return name
  return `${TOOL_NAME_PREFIX}${name}`
}

// Anthropic's tool_use.id must match ^[a-zA-Z0-9_-]+$. The harness issues
// composite ids like `call_abc123|fc_def456` (subagent / PTC dispatch) whose
// `|` is illegal on the wire. Sanitize to a legal charset, deterministically,
// so a tool_use and its tool_result always map to the same wire id.
const LEGAL_TOOL_ID = /^[a-zA-Z0-9_-]+$/
function sanitizeToolId(raw: string): string {
  if (LEGAL_TOOL_ID.test(raw)) return raw
  let digest = 5381
  for (let i = 0; i < raw.length; i++) digest = ((digest << 5) + digest + raw.charCodeAt(i)) | 0
  const suffix = Math.abs(digest).toString(36)
  return raw.replace(/[^a-zA-Z0-9_-]/g, '_') + '_' + suffix
}

/** Per-request memo so an id seen in both a tool_use and its tool_result maps identically. */
function toolIdMapper(): (raw: string) => string {
  const cache = new Map<string, string>()
  return (raw: string) => {
    let out = cache.get(raw)
    if (out === undefined) {
      out = sanitizeToolId(String(raw))
      cache.set(raw, out)
    }
    return out
  }
}

function imageBlock(ref: ImageRef, images: Map<string, ResolvedImage>): WireImageBlock | undefined {
  const resolved = images.get(ref.attachmentId)
  if (resolved === undefined) return undefined
  return { type: 'image', source: { type: 'base64', media_type: resolved.mediaType, data: resolved.dataBase64 } }
}

function serializeAssistant(message: Message, multiTool: boolean, images: Map<string, ResolvedImage>, mapId: (raw: string) => string): WireMessage | undefined {
  const text = flattenText(message.content)
  const toolCalls = message.content.filter(block => block.type === 'tool-call')
  const imageBlocks = message.content
    .filter((block): block is Extract<ContentBlock, { type: 'image' }> => block.type === 'image')
    .map(block => imageBlock(block.attachment, images))
    .filter((block): block is WireImageBlock => block !== undefined)

  const blocks: WireBlock[] = []
  if (text.trim().length > 0) blocks.push({ type: 'text', text })
  for (const img of imageBlocks) blocks.push(img)
  for (const call of toolCalls) {
    let input: unknown
    try {
      input = call.arguments === '' ? {} : JSON.parse(call.arguments)
    } catch {
      input = { _raw: call.arguments }
    }
    blocks.push({
      type: 'tool_use',
      id: mapId(call.id),
      name: prefixedToolName(call.name, multiTool),
      input,
    })
  }
  return blocks.length > 0 ? { role: 'assistant', content: blocks } : undefined
}

function toolResultContent(result: ToolResultBlock, images: Map<string, ResolvedImage>): string | WireBlock[] {
  const imagesInResult = result.content
    .filter((block): block is Extract<ContentBlock, { type: 'image' }> => block.type === 'image')
    .map(block => imageBlock(block.attachment, images))
    .filter((block): block is WireImageBlock => block !== undefined)
  const text = flattenText(result.content)
  if (imagesInResult.length === 0) return text || '(no output)'
  const blocks: WireBlock[] = []
  if (text.length > 0) blocks.push({ type: 'text', text })
  for (const img of imagesInResult) blocks.push(img)
  return blocks
}

function serializeUser(message: Message, images: Map<string, ResolvedImage>, mapId: (raw: string) => string): WireMessage[] {
  const out: WireMessage[] = []
  const userImages = message.content
    .filter((block): block is Extract<ContentBlock, { type: 'image' }> => block.type === 'image')
    .map(block => imageBlock(block.attachment, images))
    .filter((block): block is WireImageBlock => block !== undefined)
  const text = flattenText(message.content)
  const toolResults = message.content.filter((block): block is ToolResultBlock => block.type === 'tool-result')

  if (userImages.length > 0 || text.trim().length > 0) {
    if (userImages.length === 0) {
      out.push({ role: 'user', content: text })
    } else {
      const blocks: WireBlock[] = []
      if (text.trim().length > 0) blocks.push({ type: 'text', text })
      for (const img of userImages) blocks.push(img)
      out.push({ role: 'user', content: blocks })
    }
  }
  if (toolResults.length > 0) {
    const blocks: WireBlock[] = toolResults.map(result => ({
      type: 'tool_result',
      tool_use_id: mapId(result.toolCallId),
      content: toolResultContent(result, images),
      ...result.isError === true ? { is_error: true } : {},
    }))
    out.push({ role: 'user', content: blocks })
  }
  return out
}

function stripLeadingSystem(wire: WireMessage[]): WireMessage[] {
  const first = wire.find(m => m.role !== 'system')
  return first === undefined ? wire : wire.slice(wire.indexOf(first))
}

function mergeConsecutiveRoles(wire: WireMessage[]): WireMessage[] {
  const out: WireMessage[] = []
  for (const message of wire) {
    const prev = out[out.length - 1]
    if (prev !== undefined && prev.role === message.role && prev.role !== 'assistant') {
      const a = typeof prev.content === 'string' ? prev.content : ''
      const b = typeof message.content === 'string' ? message.content : ''
      if (a.length === 0) continue
      if (b.length === 0) continue
      out[out.length - 1] = { role: prev.role, content: `${a}\n\n${b}` }
      continue
    }
    out.push(message)
  }
  return out
}

function repairToolAdjacency(wire: WireMessage[]): WireMessage[] {
  // Anthropic's strict rule: every `tool_use` must be followed by a `tool_result`
  // in the IMMEDIATE next user message. Two harness behaviors break this:
  //   (a) a tool call gets interrupted (no result ever produced)
  //   (b) the harness merges adjacent user messages, so a result that was in
  //       its own message ends up in a *later* user message after the assistant
  //
  // Strategy: after the assistant, gather ALL tool_results from the following
  // user messages (up to the next assistant) and hoist the ones that pair with
  // the assistant's tool_use blocks into the immediate next user message.
  // Any tool_use without a matching result gets a placeholder.
  const out: WireMessage[] = []
  for (let i = 0; i < wire.length; i++) {
    const message = wire[i]!
    if (message.role !== 'assistant' || !Array.isArray(message.content)) {
      out.push(message)
      continue
    }
    const uses = message.content.filter((b): b is Extract<WireBlock, { type: 'tool_use' }> => b.type === 'tool_use').map(b => b.id)
    out.push(message)
    if (uses.length === 0) continue

    // Look ahead: collect tool_results from the following user messages up to
    // the next assistant (or end of wire).
    const seen = new Set<string>()
    let j = i + 1
    const collected: WireBlock[] = []
    const textCarry: string[] = []
    while (j < wire.length && wire[j]!.role === 'user') {
      const u = wire[j]!
      if (Array.isArray(u.content)) {
        for (const b of u.content) {
          if (b.type === 'tool_result') {
            if (uses.includes(b.tool_use_id) && !seen.has(b.tool_use_id)) {
              seen.add(b.tool_use_id)
              collected.push(b)
            }
          } else if (b.type === 'text' && textCarry.length === 0 && collected.length === 0) {
            // first text block of the immediate next user message — keep it inline
          }
        }
      }
      j++
    }

    // Missing results get placeholders.
    for (const id of uses) {
      if (!seen.has(id)) {
        collected.push({ type: 'tool_result', tool_use_id: id, content: 'Tool result unavailable (pruned from history).', is_error: true })
      }
    }

    // Emit the immediate next user message with the paired results. If the
    // original next user message had text, prepend it; if it had no content
    // at all, create one.
    if (i + 1 < wire.length && wire[i + 1]!.role === 'user') {
      const next = wire[i + 1]!
      const nextBlocks: WireBlock[] = Array.isArray(next.content) ? [...next.content] : (typeof next.content === 'string' && next.content.length > 0 ? [{ type: 'text', text: next.content }] : [])
      // De-duplicate: if next already has one of the collected results, don't add twice
      const existingIds = new Set(nextBlocks.filter((b): b is Extract<WireBlock, { type: 'tool_result' }> => b.type === 'tool_result').map(b => b.tool_use_id))
      const toAdd = collected.filter(b => b.type === 'tool_result' && !existingIds.has(b.tool_use_id))
      out.push({ role: 'user', content: [...toAdd, ...nextBlocks] })
      // Skip the user messages we already consumed (they were merged/looked at)
      // But we must NOT skip the immediate next — we already pushed it.
      // The look-ahead consumed j-i-1 messages total; we pushed 1 (the immediate
      // next), so skip the rest.
      i = i + 1 + (j - i - 2)
    } else {
      out.push({ role: 'user', content: collected })
    }
  }
  return out
}

export function serializeMessages(messages: readonly Message[], multiTool: boolean, images: Map<string, ResolvedImage> = new Map()): WireMessage[] {
  const mapId = toolIdMapper()
  const wire: WireMessage[] = []
  for (const message of messages) {
    if (message.role === 'system') {
      const text = flattenText(message.content)
      if (text.trim().length > 0) wire.push({ role: 'system', content: text })
      continue
    }
    if (message.role === 'assistant') {
      const assistant = serializeAssistant(message, multiTool, images, mapId)
      if (assistant !== undefined) wire.push(assistant)
      continue
    }
    for (const user of serializeUser(message, images, mapId)) wire.push(user)
  }
  return mergeConsecutiveRoles(repairToolAdjacency(stripLeadingSystem(wire)))
}

export async function serializeRequest(options: GenerateOptions, attachments?: AttachmentReader): Promise<WireRequest> {
  const multiTool = (options.tools?.length ?? 0) > 1
  const images = await resolveImages(options.messages, attachments, options.signal)
  const messages = serializeMessages(options.messages, multiTool, images)

  const systemParts: string[] = []
  if (options.system !== undefined && options.system.length > 0) systemParts.push(options.system)
  while (messages.length > 0 && messages[0].role === 'system') {
    const first = messages.shift()!
    systemParts.push(typeof first.content === 'string' ? first.content : flattenText(first.content as ContentBlock[]))
  }

  const keptSystem: string[] = []
  const movedSystem: string[] = []
  for (const part of systemParts) {
    if (part.trim().length === 0) continue
    if (part.startsWith(SYSTEM_IDENTITY)) keptSystem.push(part)
    else movedSystem.push(part)
  }

  const relocated = movedSystem.length > 0
    ? movedSystem.join('\n\n')
    : undefined
  if (relocated !== undefined) {
    const firstUserIndex = messages.findIndex(message => message.role === 'user')
    if (firstUserIndex !== -1) {
      const firstUser = messages[firstUserIndex]!
      if (typeof firstUser.content === 'string') {
        messages[firstUserIndex] = { ...firstUser, content: `${relocated}\n\n${firstUser.content}` }
      } else if (Array.isArray(firstUser.content)) {
        messages[firstUserIndex] = { ...firstUser, content: [{ type: 'text', text: relocated }, ...firstUser.content] }
      }
    }
  }

  const tools: WireTool[] | undefined = options.tools && options.tools.length > 0
    ? options.tools.map(tool => ({
        name: prefixedToolName(tool.name, multiTool),
        ...(tool.description !== undefined ? { description: tool.description } : {}),
        input_schema: (tool.parameters as Record<string, unknown>) ?? { type: 'object' },
      }))
    : undefined

  const body: Record<string, unknown> = {
    model: options.model,
    messages,
    max_tokens: options.maxTokens ?? 32_000,
    stream: true,
  }

  const firstUser = messages.find(m => m.role === 'user')
  const firstUserText = firstUser !== undefined
    ? (typeof firstUser.content === 'string'
      ? firstUser.content
      : flattenText(firstUser.content as ContentBlock[]))
    : ''
  const cliVersion = process.env.CLAUDE_CODE_VERSION ?? '2.1.268'
  const entrypoint = process.env.CLAUDE_CODE_ENTRYPOINT ?? 'sdk-cli'
  const billingText = buildBillingHeader({ firstUserMessageText: firstUserText, cliVersion, entrypoint })

  const systemBlocks: Array<{ type: 'text'; text: string }> = [
    { type: 'text', text: billingText },
    { type: 'text', text: SYSTEM_IDENTITY },
  ]
  for (const kept of keptSystem) {
    if (kept !== SYSTEM_IDENTITY) systemBlocks.push({ type: 'text', text: kept })
  }
  body.system = systemBlocks
  if (tools !== undefined) body.tools = tools
  if (options.temperature !== undefined) body.temperature = options.temperature
  if (options.stop !== undefined && options.stop.length > 0) body.stop = options.stop

  const effort = options.reasoningEffort === undefined ? undefined : String(options.reasoningEffort)
  if (effort !== undefined && effort !== 'off' && supportsEffort(options.model)) {
    body.output_config = { effort }
  }

  return body as unknown as WireRequest
}

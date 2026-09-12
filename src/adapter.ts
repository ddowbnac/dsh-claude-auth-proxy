import {
  CONTEXT_WINDOW_EXCEEDED_CODE,
  LlmError,
  ReasoningEffortId,
  type GenerateOptions,
  type LlmAdapter,
  type LlmDiscoveredModel,
  type LlmModelInfo,
  type LlmProviderInfo,
  type LlmResolvedModelInfo,
  type PreparedAdapterCall,
  type StreamChunk,
} from '@deepseek-ai/dsh-llm'
import { randomUUID } from 'node:crypto'
import { idleWatchdog, timeoutOf } from '@deepseek-ai/dsh-timeout'
import { effortLadder, getModelBetas, getNextBetaToExclude, isLongContextError, supportsEffort } from './model-config.ts'
import type { ClaudeCatalogModel } from './types.ts'
import { serializeRequest } from './serialize.ts'
import { stainlessHeaders } from './billing.ts'
import { createSseParser } from './sse.ts'
import { AnthropicEventTranslator } from './translate.ts'
import {
  ANTHROPIC_API_VERSION,
  API_BASE_URL,
  type ClaudeConnectionOptions,
} from './types.ts'
import { isUnavailableCredential, type ResolvedCredential, type UnavailableCredential } from './auth.ts'

export const PROVIDER_ID = 'claude-subscription'
export const PROVIDER_NAME = 'Claude (Claude Code subscription)'

const SESSION_ID = randomUUID()

export const DEFAULT_STREAM_IDLE_TIMEOUT_MS = 10 * 60 * 1000
export const DEFAULT_REQUEST_TIMEOUT_MS = 15 * 60 * 1000

export interface ClaudeSubscriptionAdapterOptions {
  resolveConnection: () => ClaudeConnectionOptions
  resolveCredential: (signal?: AbortSignal) => Promise<ResolvedCredential | UnavailableCredential>
  resolveModels: (signal?: AbortSignal) => Promise<ClaudeCatalogModel[]>
  resolveAttachments?: () => import('./serialize.ts').AttachmentReader | undefined
  fetchImpl?: typeof fetch
  log?: (message: string) => void
}

export class ClaudeSubscriptionAdapter implements LlmAdapter {
  private readonly excludedBetas = new Map<string, Set<string>>()

  constructor(private readonly options: ClaudeSubscriptionAdapterOptions) {}

  providerInfo(_provider: string): LlmProviderInfo {
    return { id: PROVIDER_ID, name: PROVIDER_NAME }
  }

  providerRetryPolicy(_provider: string): undefined {
    return undefined
  }

  imageRequestPricing(_provider: string, _model: string): undefined {
    return undefined
  }

  async listModels(provider: string, signal?: AbortSignal): Promise<readonly LlmModelInfo[]> {
    const models = await this.options.resolveModels(signal)
    return models.map(model => ({
      provider,
      id: model.id,
      name: model.name ?? model.id,
      ...(model.description !== undefined ? { description: model.description } : {}),
      inputModalities: ['text', 'image'],
    }))
  }

  async resolveModel(provider: string, model: string, signal?: AbortSignal): Promise<LlmResolvedModelInfo> {
    const models = await this.options.resolveModels(signal)
    const entry = models.find(m => m.id === model)
    if (entry === undefined) {
      throw new LlmError(`Unknown model for the Claude subscription adapter: ${model}`, 'NO_ADAPTER')
    }
    const efforts = entry.reasoningEfforts ?? effortLadder(entry.id)
    const defaultEffort = entry.defaultReasoningEffort
    return {
      provider,
      id: entry.id,
      name: entry.name ?? entry.id,
      ...(entry.description !== undefined ? { description: entry.description } : {}),
      inputModalities: ['text', 'image'],
      context: { contextWindow: entry.contextWindow },
      defaultMaxTokens: entry.maxTokens,
      ...(efforts !== undefined && efforts.length > 0
        ? {
            reasoning: {
              efforts: efforts.map(id => ({ id: ReasoningEffortId(id), name: id })),
              ...(defaultEffort !== undefined ? { defaultEffort: ReasoningEffortId(defaultEffort) } : {}),
            },
          }
        : {}),
    }
  }

  async prepareCall(provider: string, model: string, signal?: AbortSignal): Promise<PreparedAdapterCall> {
    const resolved = await this.resolveModel(provider, model)
    return {
      model: resolved,
      stream: (options: GenerateOptions) => this.dispatch(options, signal),
    }
  }

  async *stream(options: GenerateOptions): AsyncIterable<StreamChunk> {
    yield* this.dispatch(options, options.signal)
  }

  private async *dispatch(options: GenerateOptions, signal?: AbortSignal): AsyncIterable<StreamChunk> {
    const connection = this.options.resolveConnection()
    const credential = await this.options.resolveCredential(signal)
    if (isUnavailableCredential(credential)) {
      // No usable Claude Code credential (e.g. `claude` was never run on this
      // machine). Fail this request with a typed, actionable error instead of
      // letting the missing-credential condition surface as a fatal boot
      // error. The plugin stays up and the model catalog is served from the
      // static fallback.
      throw new LlmError(credential.reason, 'MISSING_CREDENTIAL')
    }
    const body = await serializeRequest(options, this.options.resolveAttachments?.())
    const excluded = this.excludedBetas.get(options.model) ?? new Set<string>()
    const betas = getModelBetas(options.model, excluded)

    const response = await this.send(body, betas, credential, signal, connection)
    const knownTools = new Set((options.tools ?? []).map((t) => t.name))
    yield* this.consume(response, options.model, signal, connection, knownTools)
  }

  private async send(
    body: unknown,
    betas: string[],
    credential: ResolvedCredential,
    signal: AbortSignal | undefined,
    connection: ClaudeConnectionOptions,
  ): Promise<Response> {
    const controller = new AbortController()
    const onAbort = () => controller.abort(signal?.reason)
    signal?.addEventListener('abort', onAbort, { once: true })
    try {
      const headers = this.buildHeaders(betas, credential)
      return await (this.options.fetchImpl ?? fetch)(`${API_BASE_URL}/v1/messages?beta=true`, {
        method: 'POST',
        headers,
        body: JSON.stringify(body),
        signal: controller.signal,
      })
    } finally {
      signal?.removeEventListener('abort', onAbort)
    }
  }

  private buildHeaders(betas: string[], credential: ResolvedCredential): Headers {
    const headers = new Headers()
    headers.set('authorization', `Bearer ${credential.accessToken}`)
    headers.set('anthropic-version', ANTHROPIC_API_VERSION)
    headers.set('anthropic-beta', betas.join(','))
    headers.set('anthropic-dangerous-direct-browser-access', 'true')
    headers.set('x-app', 'cli')
    headers.set('user-agent', `claude-cli/${this.cliVersion()} (external, sdk-cli)`)
    headers.set('content-type', 'application/json')
    headers.set('x-client-request-id', randomUUID())
    headers.set('x-claude-code-session-id', SESSION_ID)
    for (const [key, value] of Object.entries(stainlessHeaders())) {
      if (!headers.has(key)) headers.set(key, value)
    }
    return headers
  }

  private cliVersion(): string {
    return process.env.CLAUDE_CODE_VERSION ?? '2.1.268'
  }

  private async *consume(response: Response, modelId: string, signal: AbortSignal | undefined, connection: ClaudeConnectionOptions, knownTools: ReadonlySet<string>): AsyncIterable<StreamChunk> {
    if (!response.ok) {
      const text = await response.text().catch(() => '')
      if (isLongContextError(text)) {
        const excluded = this.excludedBetas.get(modelId) ?? new Set<string>()
        const next = getNextBetaToExclude(modelId, excluded)
        if (next !== undefined) {
          excluded.add(next)
          this.excludedBetas.set(modelId, excluded)
          throw new LlmError('Long-context beta not available for this plan; continuing with a reduced beta set', 'QUOTA', {
            status: response.status,
          })
        }
      }
      throw this.errorFromHttpResponse(response.status, text)
    }
    if (!response.body) {
      throw new LlmError('Claude returned no response body', 'EMPTY_RESPONSE')
    }

    const parser = createSseParser()
    const translator = new AnthropicEventTranslator(knownTools)
    const reader = response.body.getReader()
    const watchdog = idleWatchdog(signal, connection.streamIdleTimeoutMs, 'STREAM_IDLE_TIMEOUT')
    try {
      for (;;) {
        const { done, value } = await watchdog.next({
          next: () => reader.read(),
        })
        if (done || value === undefined) break
        for (const event of parser.push(value)) {
          yield* translator.translate(event)
        }
      }
      for (const event of parser.end()) {
        yield* translator.translate(event)
      }
    } catch (error) {
      const timeout = timeoutOf(error as { reason?: unknown }, 'STREAM_IDLE_TIMEOUT')
      if (timeout !== undefined) {
        yield { type: 'finish', reason: { kind: 'error', failure: { message: timeout.message, code: 'TIMEOUT' } } }
        return
      }
      const aborted = signal?.aborted === true
      const message = error instanceof Error ? error.message : String(error)
      const code = aborted ? 'ABORTED' : 'TRANSPORT'
      yield { type: 'finish', reason: { kind: aborted ? 'aborted' : 'error', failure: { message, code } } }
      return
    } finally {
      watchdog[Symbol.dispose]()
    }
  }

  private errorFromHttpResponse(status: number, text: string): LlmError {
    let type: string | undefined
    let message = `Claude API error (HTTP ${status})`
    try {
      const parsed = JSON.parse(text) as Record<string, unknown>
      if (typeof parsed.error === 'object' && parsed.error !== null) {
        const err = parsed.error as Record<string, unknown>
        type = typeof err.type === 'string' ? err.type : undefined
        message = typeof err.message === 'string' ? err.message : message
      }
    } catch {
    }
    const detail = `${type ?? ''} ${message}`.toLowerCase()
    let code: string
    if (status === 401) code = 'AUTH'
    else if (status === 403) code = 'FORBIDDEN'
    else if (isQuotaWording(detail)) code = 'QUOTA'
    else if (status === 429) code = 'RATE_LIMIT'
    else if (status >= 500) code = 'SERVER'
    else if (isContextOverflowWording(detail)) code = CONTEXT_WINDOW_EXCEEDED_CODE
    else code = type ?? 'API_ERROR'
    return new LlmError(message, code, { status })
  }

  dispose(): void {
  }
}

function isContextOverflowWording(detail: string): boolean {
  return /context(?:[\s_-]?(?:length|window))?(?:[\s_-]?exceeded|overflow|too large|too long)/i.test(detail)
    || /prompt is too (?:long|large)/i.test(detail)
    || /maximum context/i.test(detail)
}

function isQuotaWording(detail: string): boolean {
  return /out of (?:extra )?usage|usage limit exceeded|exceeded your (?:quota|limit)|quota/i.test(detail)
}

export { supportsEffort }
export type { LlmDiscoveredModel }

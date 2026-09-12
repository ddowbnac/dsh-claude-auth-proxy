import { createHash, randomUUID } from 'node:crypto'
import { LlmError } from '@deepseek-ai/dsh-llm'
import { existsSync, readFileSync, writeFileSync, renameSync, unlinkSync, openSync, closeSync, chmodSync, constants, statSync } from 'node:fs'
import { homedir } from 'node:os'
import { basename, dirname, join, isAbsolute } from 'node:path'
import {
  ANTHROPIC_API_VERSION,
  API_BASE_URL,
  OAUTH_CLIENT_ID,
  OAUTH_TOKEN_URL,
  type ClaudeOAuthCredentials,
} from './types.ts'

export interface DiscoveredModel {
  id: string
  name: string
  contextWindow: number
  maxTokens: number
}

interface WireModel {
  id?: string
  display_name?: string
  max_input_tokens?: number
  max_tokens?: number
}

const OAUTH_TIMEOUT_MS = 15_000
const MAX_RETRY_DELAY_MS = 60_000
const RETRY_BASE_DELAY_MS = 2_000
const RETRYABLE_REFRESH_STATUSES = new Set([429, 500, 502, 503, 504])
const CACHE_TTL_MS = 30_000
const LOCK_WAIT_TIMEOUT_MS = 15_000
const LOCK_WAIT_POLL_MS = 500
const MODELS_CACHE_TTL_MS = 5 * 60 * 1000

export type RefreshOutcome =
  | { kind: 'ok'; credentials: ClaudeOAuthCredentials }
  | { kind: 'transient'; status: number; detail?: string; retryAfterMs?: number }
  | { kind: 'terminal'; status: number; detail?: string }
  | { kind: 'none' }

export interface ResolvedCredential {
  accessToken: string
  source: string
}

function expandHome(p: string): string {
  if (p === '~') return homedir()
  if (p.startsWith('~/')) return join(homedir(), p.slice(2))
  return p
}

function defaultCredentialsPath(): string {
  return join(homedir(), '.claude', '.credentials.json')
}

export function parseClaudeAiOauth(raw: string): ClaudeOAuthCredentials | null {
  let doc: unknown
  try {
    doc = JSON.parse(raw)
  } catch {
    return null
  }
  if (typeof doc !== 'object' || doc === null) return null
  const entry = (doc as Record<string, unknown>)['claudeAiOauth']
  if (typeof entry !== 'object' || entry === null) return null
  const e = entry as Record<string, unknown>
  if (typeof e.accessToken !== 'string' || e.accessToken.length === 0) return null
  if (typeof e.refreshToken !== 'string' || e.refreshToken.length === 0) return null
  const expiresAt = typeof e.expiresAt === 'number' ? e.expiresAt : 0
  return {
    accessToken: e.accessToken,
    refreshToken: e.refreshToken,
    expiresAt,
    ...(typeof e.refreshTokenExpiresAt === 'number' ? { refreshTokenExpiresAt: e.refreshTokenExpiresAt } : {}),
    ...(Array.isArray(e.scopes) ? { scopes: e.scopes.map(String) } : {}),
    ...(typeof e.subscriptionType === 'string' ? { subscriptionType: e.subscriptionType } : {}),
  }
}

export function writeBackCredentials(path: string, credentials: ClaudeOAuthCredentials): boolean {
  let doc: Record<string, unknown>
  try {
    const current = readFileSync(path, 'utf8')
    doc = JSON.parse(current) as Record<string, unknown>
  } catch {
    return false
  }
  if (typeof doc !== 'object' || doc === null) return false
  const previous = parseClaudeAiOauth(JSON.stringify(doc))
  if (previous !== null && previous.refreshToken !== credentials.refreshToken) {
    return false
  }
  doc.claudeAiOauth = {
    ...credentials,
    scopes: credentials.scopes ?? [],
  }
  const target = join(dirname(path), `.${basename(path).replace(/\.[^.]*$/, '')}.${randomUUID()}.tmp`)
  writeFileSync(target, JSON.stringify(doc, null, 2), { mode: 0o600 })
  try {
    renameSync(target, path)
  } catch (error) {
    try {
      unlinkSync(target)
    } catch {
    }
    throw error
  }
  try {
    chmodSync(path, 0o600)
  } catch {
  }
  return true
}

function parseRetryAfterMs(headerValue: string | null): number | undefined {
  if (!headerValue) return undefined
  const seconds = Number.parseInt(headerValue, 10)
  return Number.isFinite(seconds) && seconds > 0 ? Math.min(seconds * 1000, MAX_RETRY_DELAY_MS) : undefined
}

function extractOAuthError(raw: string): string | undefined {
  let data: unknown
  try {
    data = JSON.parse(raw)
  } catch {
    return undefined
  }
  if (typeof data !== 'object' || data === null) return undefined
  const d = data as Record<string, unknown>
  if (typeof d.error === 'string') return d.error.slice(0, 200)
  if (typeof d.error === 'object' && d.error !== null) {
    const nested = d.error as Record<string, unknown>
    if (typeof nested.message === 'string') return nested.message.slice(0, 200)
    if (typeof nested.type === 'string') return nested.type.slice(0, 200)
  }
  if (typeof d.error_description === 'string') return d.error_description.slice(0, 200)
  return undefined
}

export async function refreshViaOAuth(
  refreshToken: string,
  fetchImpl: typeof fetch = fetch,
): Promise<RefreshOutcome> {
  const body = new URLSearchParams({
    grant_type: 'refresh_token',
    client_id: OAUTH_CLIENT_ID,
    refresh_token: refreshToken,
  })
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), OAUTH_TIMEOUT_MS)
  try {
    const response = await fetchImpl(OAUTH_TOKEN_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: body.toString(),
      signal: controller.signal,
    })
    if (!response.ok) {
      const detail = extractOAuthError(await response.text().catch(() => ''))
      const retryAfterMs = parseRetryAfterMs(response.headers.get('retry-after'))
      const kind: RefreshOutcome['kind'] = RETRYABLE_REFRESH_STATUSES.has(response.status) || response.status === 429
        ? 'transient'
        : 'terminal'
      return { kind, status: response.status, detail, ...retryAfterMs !== undefined ? { retryAfterMs } : {} }
    }
    const text = await response.text()
    let data: Record<string, unknown>
    try {
      data = JSON.parse(text)
    } catch {
      return { kind: 'transient', status: response.status, detail: 'unparseable refresh response' }
    }
    const accessToken = typeof data.access_token === 'string' ? data.access_token : undefined
    if (accessToken === undefined) {
      return { kind: 'transient', status: response.status, detail: 'refresh response carried no access_token' }
    }
    const now = Date.now()
    const expiresAt = typeof data.expires_at === 'number' && data.expires_at > now
      ? Math.trunc(data.expires_at)
      : Math.trunc(now + (typeof data.expires_in === 'number' ? data.expires_in : 36_000) * 1000)
    const credentials: ClaudeOAuthCredentials = {
      accessToken,
      refreshToken: typeof data.refresh_token === 'string' ? data.refresh_token : refreshToken,
      expiresAt,
    }
    return { kind: 'ok', credentials }
  } catch (error) {
    const aborted = controller.signal.aborted
    return {
      kind: 'transient',
      status: 0,
      detail: aborted ? 'refresh timed out' : error instanceof Error ? error.message : String(error),
    }
  } finally {
    clearTimeout(timer)
  }
}

interface CacheEntry {
  accessToken: string
  expiresAt: number
  cachedAt: number
  source: string
}

export class ClaudeCredentialStore {
  private readonly path: string
  private refreshInFlight: Promise<RefreshOutcome> | undefined
  private backoffUntil = 0
  private proactiveTimer: ReturnType<typeof setInterval> | undefined
  private backoffLadder = 0
  private cache: CacheEntry | undefined

  constructor(path: string | undefined) {
    this.path = isAbsolute(expandHome(path ?? '')) ? expandHome(path ?? '') : defaultCredentialsPath()
  }

  get credentialsPath(): string {
    return this.path
  }

  read(): ClaudeOAuthCredentials | null {
    try {
      if (!existsSync(this.path)) return null
      return parseClaudeAiOauth(readFileSync(this.path, 'utf8'))
    } catch {
      return null
    }
  }

  startProactiveRefresh(intervalMs = 5 * 60 * 1000): void {
    if (this.proactiveTimer !== undefined) return
    const tick = () => {
      void this.resolve(undefined, 60 * 60 * 1000).catch(() => undefined)
    }
    this.proactiveTimer = setInterval(tick, intervalMs)
    this.proactiveTimer.unref()
    void this.resolve(undefined, 60 * 60 * 1000).catch(() => undefined)
  }

  stop(): void {
    if (this.proactiveTimer !== undefined) clearInterval(this.proactiveTimer)
    this.proactiveTimer = undefined
  }

  async resolve(signal?: AbortSignal, thresholdMs = 0): Promise<ResolvedCredential> {
    const now = Date.now()
    if (this.cache !== undefined && now - this.cache.cachedAt < CACHE_TTL_MS && this.cache.expiresAt > now + thresholdMs) {
      return { accessToken: this.cache.accessToken, source: this.cache.source }
    }
    const creds = this.read()
    if (creds === null) {
      throw new LlmError(
        this.fileMissing()
          ? `no Claude Code credentials file at ${this.path} — run \`claude\` once to authenticate`
          : `no Claude Code OAuth entry in ${this.path} — authenticate with a Claude subscription (\`claude\` → log in)`,
        'MISSING_CREDENTIAL',
      )
    }
    if (creds.expiresAt > now + thresholdMs) {
      this.cache = { accessToken: creds.accessToken, expiresAt: creds.expiresAt, cachedAt: now, source: `oauth(${this.path})` }
      this.resetBackoff()
      return { accessToken: creds.accessToken, source: this.cache.source }
    }
    if (now < this.backoffUntil) {
      const waitMs = Math.min(this.backoffUntil - now, MAX_RETRY_DELAY_MS)
      await abortableDelay(waitMs, signal)
    }
    const refreshed = await this.refresh(signal)
    if (refreshed.kind === 'none') {
      throw new LlmError(
        this.fileMissing()
          ? `no Claude Code credentials file at ${this.path} — run \`claude\` once to authenticate`
          : `no Claude Code OAuth entry in ${this.path} — authenticate with a Claude subscription (\`claude\` → log in)`,
        'MISSING_CREDENTIAL',
      )
    }
    if (refreshed.kind === 'transient' || refreshed.kind === 'terminal') {
      const adopted = this.read()
      if (adopted !== null && adopted.expiresAt > Date.now()) {
        this.cache = { accessToken: adopted.accessToken, expiresAt: adopted.expiresAt, cachedAt: Date.now(), source: `oauth(${this.path})` }
        return { accessToken: adopted.accessToken, source: this.cache.source }
      }
      this.applyBackoff(refreshed)
      const detail = refreshed.detail ?? `HTTP ${refreshed.status}`
      throw refreshed.kind === 'terminal'
        ? new LlmError(
          `Claude Code OAuth refresh rejected (${detail}); the refresh token may be dead — run \`claude\` to re-authenticate`,
          'INVALID_CREDENTIAL',
        )
        : new LlmError(
          `Claude Code OAuth refresh is failing (${detail}); retrying after backoff`,
          'RATE_LIMIT',
        )
    }
    this.resetBackoff()
    const fallback = this.read()
    const chosen = fallback !== null && fallback.expiresAt > Date.now() ? fallback : refreshed.credentials
    this.cache = { accessToken: chosen.accessToken, expiresAt: chosen.expiresAt, cachedAt: Date.now(), source: `oauth(${this.path})` }
    return { accessToken: chosen.accessToken, source: this.cache.source }
  }

  private invalidateCache(): void {
    this.cache = undefined
  }

  private fileMissing(): boolean {
    try {
      return !existsSync(this.path)
    } catch {
      return true
    }
  }

  private resetBackoff(): void {
    this.backoffUntil = 0
    this.backoffLadder = 0
  }

  private applyBackoff(outcome: Extract<RefreshOutcome, { kind: 'transient' | 'terminal' }>): void {
    if (outcome.kind === 'terminal') {
      this.backoffUntil = Date.now() + RETRY_BASE_DELAY_MS
      this.backoffLadder = 0
      return
    }
    const ms = outcome.retryAfterMs ?? Math.min(
      RETRY_BASE_DELAY_MS * 2 ** this.backoffLadder,
      MAX_RETRY_DELAY_MS,
    )
    this.backoffLadder = Math.min(this.backoffLadder + 1, 30)
    this.backoffUntil = Date.now() + ms
  }

  private refresh(signal?: AbortSignal): Promise<RefreshOutcome> {
    if (this.refreshInFlight !== undefined) return this.refreshInFlight
    const attempt = this.doRefresh(signal)
    this.refreshInFlight = attempt
    void attempt.finally(() => {
      if (this.refreshInFlight === attempt) this.refreshInFlight = undefined
    })
    return attempt
  }

  private modelsCache: { models: DiscoveredModel[]; cachedAt: number } | undefined
  private modelsInFlight: Promise<DiscoveredModel[]> | undefined

  async discoverModels(signal?: AbortSignal): Promise<DiscoveredModel[]> {
    const now = Date.now()
    if (this.modelsCache !== undefined && now - this.modelsCache.cachedAt < MODELS_CACHE_TTL_MS) {
      return this.modelsCache.models
    }
    if (this.modelsInFlight !== undefined) return this.modelsInFlight
    const attempt = (async (): Promise<DiscoveredModel[]> => {
      try {
        const credential = await this.resolve(signal, 0)
        const response = await fetch(`${API_BASE_URL}/v1/models`, {
          headers: {
            authorization: `Bearer ${credential.accessToken}`,
            'anthropic-version': ANTHROPIC_API_VERSION,
            'user-agent': 'claude-cli/2.1.268 (external, sdk-cli)',
            'x-app': 'cli',
            'content-type': 'application/json',
          },
          signal,
        })
        if (!response.ok) {
          const error = new LlmError(`model discovery failed (HTTP ${response.status})`, 'API_ERROR', { status: response.status })
          throw error
        }
        const doc = (await response.json()) as { data?: WireModel[] }
        const models: DiscoveredModel[] = []
        for (const entry of doc.data ?? []) {
          if (typeof entry.id !== 'string' || entry.id.length === 0) continue
          models.push({
            id: entry.id,
            name: typeof entry.display_name === 'string' && entry.display_name.length > 0 ? entry.display_name : entry.id,
            contextWindow: typeof entry.max_input_tokens === 'number' ? entry.max_input_tokens : 200_000,
            maxTokens: typeof entry.max_tokens === 'number' ? entry.max_tokens : 64_000,
          })
        }
        if (models.length === 0) {
          const error = new LlmError('model discovery returned an empty list', 'API_ERROR')
          throw error
        }
        this.modelsCache = { models, cachedAt: Date.now() }
        return models
      } catch (error) {
        if (this.modelsCache !== undefined) return this.modelsCache.models
        throw error
      }
    })()
    this.modelsInFlight = attempt
    void attempt.finally(() => {
      if (this.modelsInFlight === attempt) this.modelsInFlight = undefined
    })
    return attempt
  }

  private async doRefresh(signal?: AbortSignal): Promise<RefreshOutcome> {
    if (signal?.aborted) return { kind: 'none' }
    const creds = this.read()
    if (creds === null) return { kind: 'none' }
    if (typeof creds.refreshToken !== 'string' || creds.refreshToken.length === 0) return { kind: 'none' }
    if (typeof creds.refreshTokenExpiresAt === 'number' && creds.refreshTokenExpiresAt < Date.now()) {
      return { kind: 'terminal', status: 400, detail: 'refresh token itself expired — run `claude` to re-authenticate' }
    }
    const lock = acquireCrossProcessLock(this.path)
    if (lock === '') {
      const adopted = await waitForAdopt(this.path, LOCK_WAIT_TIMEOUT_MS, signal)
      if (adopted) return { kind: 'ok', credentials: adopted }
      return { kind: 'transient', status: 429, detail: 'another process is refreshing' }
    }
    try {
      const rechecked = this.read()
      if (rechecked !== null && rechecked.expiresAt > Date.now()) return { kind: 'ok', credentials: rechecked }
      const toRefresh = rechecked ?? creds
      const outcome = await refreshViaOAuth(toRefresh.refreshToken)
      if (outcome.kind === 'ok') {
        const written = writeBackCredentials(this.path, outcome.credentials)
        if (!written) {
          const after = this.read()
          if (after !== null && after.expiresAt > Date.now()) return { kind: 'ok', credentials: after }
        }
      }
      return outcome
    } finally {
      releaseCrossProcessLock(this.path, lock)
    }
  }
}

async function waitForAdopt(path: string, timeoutMs: number, signal?: AbortSignal): Promise<ClaudeOAuthCredentials | null> {
  const deadline = Date.now() + timeoutMs
  for (;;) {
    if (signal?.aborted) return null
    try {
      const fresh = parseClaudeAiOauth(readFileSync(path, 'utf8'))
      if (fresh !== null && fresh.expiresAt > Date.now()) return fresh
    } catch {
    }
    if (Date.now() >= deadline) return null
    await abortableDelay(LOCK_WAIT_POLL_MS, signal).catch(() => undefined)
  }
}

function abortableDelay(ms: number, signal?: AbortSignal): Promise<void> {
  if (ms <= 0) return Promise.resolve()
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => cleanup(resolve), ms)
    const onAbort = () => {
      cleanup(() => reject(new LlmError('aborted', 'ABORTED')))
    }
    const cleanup = (finish: () => void) => {
      clearTimeout(timer)
      signal?.removeEventListener('abort', onAbort)
      finish()
    }
    signal?.addEventListener('abort', onAbort, { once: true })
  })
}

const lockPathFor = (path: string): string => `${path}.refresh-lock`
const LOCK_TTL_MS = 20_000

interface LockRecord {
  pid: number
  ts: number
}

function isProcessAlive(pid: number): boolean {
  try {
    process.kill(pid, 0)
    return true
  } catch (error) {
    return (error as NodeJS.ErrnoException)?.code === 'EPERM'
  }
}

function acquireCrossProcessLock(path: string): string {
  const lockPath = lockPathFor(path)
  const claim = (): boolean => {
    try {
      const fd = openSync(lockPath, constants.O_WRONLY | constants.O_CREAT | constants.O_EXCL, 0o600)
      closeSync(fd)
      const record: LockRecord = { pid: process.pid, ts: Date.now() }
      writeFileSync(lockPath, JSON.stringify(record))
      return true
    } catch {
      return false
    }
  }
  if (claim()) return lockPath
  try {
    const ageMs = Date.now() - statSync(lockPath).mtimeMs
    let record: LockRecord | undefined
    try {
      record = JSON.parse(readFileSync(lockPath, 'utf8')) as LockRecord
    } catch {
    }
    const holderDead = record?.pid !== undefined && !isProcessAlive(record.pid)
    if (ageMs > LOCK_TTL_MS || holderDead) {
      try {
        unlinkSync(lockPath)
      } catch {
      }
      return claim() ? lockPath : ''
    }
    return ''
  } catch {
    return ''
  }
}

function releaseCrossProcessLock(path: string, lock: string): void {
  if (lock === '') return
  try {
    unlinkSync(lock)
  } catch {
  }
}

export function credentialsFingerprint(path: string): string | undefined {
  try {
    const raw = readFileSync(path, 'utf8')
    return createHash('sha256').update(raw).digest('hex').slice(0, 16)
  } catch {
    return undefined
  }
}

import { afterEach, beforeEach, describe, expect, test } from 'bun:test'
import { mkdtempSync, readFileSync, writeFileSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import {
  ClaudeCredentialStore,
  isUnavailableCredential,
  parseClaudeAiOauth,
  refreshViaOAuth,
  writeBackCredentials,
} from '../src/auth.ts'

const FUTURE = Date.now() + 24 * 60 * 60 * 1000
const PAST = Date.now() - 60 * 1000

function writeCreds(path: string, doc: Record<string, unknown>): void {
  writeFileSync(path, JSON.stringify(doc, null, 2))
}

describe('parseClaudeAiOauth', () => {
  test('extracts the claudeAiOauth entry', () => {
    const creds = parseClaudeAiOauth(JSON.stringify({
      claudeAiOauth: { accessToken: 'a', refreshToken: 'r', expiresAt: FUTURE },
      mcpOAuth: { something: 'else' },
    }))
    expect(creds).toEqual({ accessToken: 'a', refreshToken: 'r', expiresAt: FUTURE })
  })

  test('returns null for malformed documents', () => {
    expect(parseClaudeAiOauth('not json')).toBeNull()
    expect(parseClaudeAiOauth(JSON.stringify({}))).toBeNull()
    expect(parseClaudeAiOauth(JSON.stringify({ claudeAiOauth: { accessToken: 'a' } }))).toBeNull()
  })
})

describe('writeBackCredentials', () => {
  let dir: string
  let path: string
  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), 'dsh-ca-'))
    path = join(dir, '.credentials.json')
  })
  afterEach(() => rmSync(dir, { recursive: true, force: true }))

  test('writes the refreshed entry, preserving other keys', () => {
    writeCreds(path, {
      claudeAiOauth: { accessToken: 'old', refreshToken: 'r1', expiresAt: PAST },
      mcpOAuth: { kept: true },
    })
    const ok = writeBackCredentials(path, { accessToken: 'new', refreshToken: 'r1', expiresAt: FUTURE })
    expect(ok).toBe(true)
    const doc = JSON.parse(readFileSync(path, 'utf8')) as Record<string, unknown>
    expect(doc.mcpOAuth).toEqual({ kept: true })
    const entry = doc.claudeAiOauth as Record<string, unknown>
    expect(entry.accessToken).toBe('new')
    expect(entry.refreshToken).toBe('r1')
  })

  test('returns false when someone else rotated the refresh token', () => {
    writeCreds(path, {
      claudeAiOauth: { accessToken: 'old', refreshToken: 'r1', expiresAt: PAST },
    })
    writeCreds(path, {
      claudeAiOauth: { accessToken: 'theirs', refreshToken: 'r-rotated', expiresAt: FUTURE },
    })
    const ok = writeBackCredentials(path, { accessToken: 'new', refreshToken: 'r1', expiresAt: FUTURE })
    expect(ok).toBe(false)
    const doc = JSON.parse(readFileSync(path, 'utf8')) as Record<string, unknown>
    expect((doc.claudeAiOauth as Record<string, unknown>).accessToken).toBe('theirs')
  })
})

describe('refreshViaOAuth', () => {
  test('classifies an ok response', async () => {
    const fetchImpl = (async () => new Response(JSON.stringify({
      access_token: 'new',
      expires_in: 3600,
    }), { status: 200 })) as unknown as typeof fetch
    const outcome = await refreshViaOAuth('r', fetchImpl)
    expect(outcome.kind).toBe('ok')
    if (outcome.kind === 'ok') {
      expect(outcome.credentials.accessToken).toBe('new')
      expect(outcome.credentials.expiresAt).toBeGreaterThan(Date.now())
    }
  })

  test('classifies 429 as transient with retry-after', async () => {
    const fetchImpl = (async () => new Response(JSON.stringify({ error: 'rate_limit_error' }), {
      status: 429,
      headers: { 'retry-after': '30' },
    })) as unknown as typeof fetch
    const outcome = await refreshViaOAuth('r', fetchImpl)
    expect(outcome.kind).toBe('transient')
    if (outcome.kind === 'transient') expect(outcome.retryAfterMs).toBe(30_000)
  })

  test('classifies invalid_grant as terminal', async () => {
    const fetchImpl = (async () => new Response(JSON.stringify({ error: 'invalid_grant' }), {
      status: 400,
    })) as unknown as typeof fetch
    const outcome = await refreshViaOAuth('r', fetchImpl)
    expect(outcome.kind).toBe('terminal')
  })
})

describe('ClaudeCredentialStore', () => {
  let dir: string
  let path: string

  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), 'dsh-cs-'))
    path = join(dir, '.credentials.json')
  })
  afterEach(() => rmSync(dir, { recursive: true, force: true }))

  test('resolves a fresh token without refreshing', async () => {
    writeCreds(path, { claudeAiOauth: { accessToken: 'fresh', refreshToken: 'r', expiresAt: FUTURE } })
    const store = new ClaudeCredentialStore(path)
    const resolved = await store.resolve()
    expect(isUnavailableCredential(resolved)).toBe(false)
    if (!isUnavailableCredential(resolved)) {
      expect(resolved.accessToken).toBe('fresh')
    }
  })

  test('refreshes an expired token via fetch', async () => {
    writeCreds(path, { claudeAiOauth: { accessToken: 'stale', refreshToken: 'r1', expiresAt: PAST } })
    const fetchImpl = (async () => new Response(JSON.stringify({
      access_token: 'rotated',
      refresh_token: 'r1',
      expires_in: 3600,
    }), { status: 200 })) as unknown as typeof fetch
    const realFetch = globalThis.fetch
    globalThis.fetch = fetchImpl
    try {
      const store = new ClaudeCredentialStore(path)
      const resolved = await store.resolve()
      expect(isUnavailableCredential(resolved)).toBe(false)
      if (!isUnavailableCredential(resolved)) {
        expect(resolved.accessToken).toBe('rotated')
      }
      const doc = JSON.parse(readFileSync(path, 'utf8')) as Record<string, unknown>
      expect((doc.claudeAiOauth as Record<string, unknown>).accessToken).toBe('rotated')
    } finally {
      globalThis.fetch = realFetch
    }
  })

  test('returns an unavailable marker (not a throw) when no credentials exist', async () => {
    const store = new ClaudeCredentialStore(path)
    const resolved = await store.resolve()
    expect(isUnavailableCredential(resolved)).toBe(true)
    if (isUnavailableCredential(resolved)) {
      expect(resolved.reason).toMatch(/no Claude Code credentials file/)
      expect(resolved.reason).toMatch(/run `claude` once to authenticate/)
    }
  })

  test('returns an unavailable marker when the file exists but has no OAuth entry', async () => {
    writeCreds(path, { someOtherProvider: { accessToken: 'x' } })
    const store = new ClaudeCredentialStore(path)
    const resolved = await store.resolve()
    expect(isUnavailableCredential(resolved)).toBe(true)
    if (isUnavailableCredential(resolved)) {
      expect(resolved.reason).toMatch(/no Claude Code OAuth entry/)
    }
  })

  test('fails terminally when the refresh token itself is expired', async () => {
    writeCreds(path, {
      claudeAiOauth: { accessToken: 'stale', refreshToken: 'r', expiresAt: PAST, refreshTokenExpiresAt: PAST },
    })
    const store = new ClaudeCredentialStore(path)
    await expect(store.resolve()).rejects.toThrow(/refresh token/i)
  })

  test('discoverModels returns an empty list (no throw) when credentials are missing', async () => {
    // Regression: boot-time model catalog construction must not be fatal when
    // the machine has never run `claude`. The caller falls back to the static
    // catalog; the first real request surfaces the typed error instead.
    const store = new ClaudeCredentialStore(path)
    const models = await store.discoverModels()
    expect(models).toEqual([])
  })
})

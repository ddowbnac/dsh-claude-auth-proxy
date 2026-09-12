import { afterEach, beforeEach, describe, expect, test } from 'bun:test'
import { mkdtempSync, readFileSync, writeFileSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { ClaudeCredentialStore, isUnavailableCredential, writeBackCredentials } from '../src/auth.ts'

const FUTURE = Date.now() + 24 * 60 * 60 * 1000
const PAST = Date.now() - 60 * 1000

function writeCreds(path: string, doc: Record<string, unknown>): void {
  writeFileSync(path, JSON.stringify(doc, null, 2))
}

describe('lock contention', () => {
  let dir: string
  let path: string

  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), 'dsh-lk-'))
    path = join(dir, '.credentials.json')
  })
  afterEach(() => rmSync(dir, { recursive: true, force: true }))

  test('second store waits and adopts the winner token instead of double-refreshing', async () => {
    writeCreds(path, {
      claudeAiOauth: { accessToken: 'old', refreshToken: 'r1', expiresAt: PAST },
    })
    let refreshCalls = 0
    const realFetch = globalThis.fetch
    globalThis.fetch = (async (input: Parameters<typeof fetch>[0], init?: RequestInit) => {
      const url = String(input)
      if (url.includes('oauth/token')) {
        refreshCalls++
        await new Promise(r => setTimeout(r, 300))
        return new Response(JSON.stringify({
          access_token: 'fresh-from-winner',
          refresh_token: 'r1',
          expires_in: 3600,
        }), { status: 200 })
      }
      return realFetch(input, init)
    }) as unknown as typeof fetch

    try {
      const lockPath = `${path}.refresh-lock`
      const { openSync, closeSync, writeFileSync: wfs, rmSync: rfs } = await import('node:fs')
      const fd = openSync(lockPath, 'w')
      closeSync(fd)
      wfs(lockPath, JSON.stringify({ pid: process.pid, ts: Date.now() }))

      const store = new ClaudeCredentialStore(path)
      const race = store.resolve().then(r => ({ ok: isUnavailableCredential(r) ? undefined : r.accessToken }))

      await new Promise(r => setTimeout(r, 100))
      const winner = { accessToken: 'fresh-from-winner', refreshToken: 'r1', expiresAt: Date.now() + 3600 * 1000 }
      writeBackCredentials(path, winner)
      rfs(lockPath)

      const result = await race
      expect(result.ok).toBe('fresh-from-winner')
      expect(refreshCalls).toBe(0)
    } finally {
      globalThis.fetch = realFetch
    }
  })

  test('cache serves within TTL without re-reading the file', async () => {
    writeCreds(path, {
      claudeAiOauth: { accessToken: 'cached-token', refreshToken: 'r1', expiresAt: FUTURE },
    })
    const store = new ClaudeCredentialStore(path)
    const first = await store.resolve()
    expect(isUnavailableCredential(first)).toBe(false)
    if (!isUnavailableCredential(first)) {
      expect(first.accessToken).toBe('cached-token')
    }
    const second = await store.resolve()
    expect(isUnavailableCredential(second)).toBe(false)
    if (!isUnavailableCredential(second)) {
      expect(second.accessToken).toBe('cached-token')
    }
  })
})

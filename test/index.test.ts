import { afterEach, beforeEach, describe, expect, test } from 'bun:test'
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import * as plugin from '../src/index.ts'
import { PROVIDER_ID, PROVIDER_NAME } from '../src/adapter.ts'

function makeCtx() {
  const calls: Record<string, unknown> = {}
  const ctx = {
    logger: { info: () => undefined, warn: () => undefined, error: () => undefined },
    llm: {
      registerAdapter: (providers: string[], adapter: unknown) => {
        calls.registerAdapter = { providers, adapter }
        return () => undefined
      },
      registerConfigurableProviders: (entries: unknown[]) => {
        calls.directory = entries
        return () => undefined
      },
    },
    inject: (deps: string[], cb: (c: { settings: { installSection: (...a: unknown[]) => void } }) => void) => {
      calls.injected = deps
      cb({
        settings: {
          installSection: (...args: unknown[]) => {
            calls.section = args
            const hooks = args[4] as { setSource?: (s: unknown) => void }
            hooks.setSource?.(() => args[3])
          },
        },
      })
    },
    effect: (fn: () => () => void, label?: string) => {
      calls.effect = label
      fn
    },
  }
  return { ctx, calls }
}

describe('plugin entry', () => {
  test('exports name, inject and Config', () => {
    expect(plugin.name).toBe('claude-auth')
    expect(plugin.inject).toEqual(['llm'])
    expect(plugin.Config).toBeDefined()
  })

  // apply() reads the credentials file when it builds the provider directory
  // entry. Pointing at a fixture (instead of the default ~/.claude/.credentials.json)
  // keeps the assertions identical on machines with a real `claude` login and
  // on CI runners without one.
  let dir: string
  let credsPath: string
  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), 'dsh-plugin-'))
    credsPath = join(dir, '.credentials.json')
    writeFileSync(credsPath, JSON.stringify({
      claudeAiOauth: { accessToken: 'a', refreshToken: 'r', expiresAt: Date.now() + 3_600_000 },
    }))
  })
  afterEach(() => rmSync(dir, { recursive: true, force: true }))

  test('apply registers the claude-subscription provider and a settings section', () => {
    const { ctx, calls } = makeCtx()
    const config = plugin.Config({ credentialsPath: credsPath })
    plugin.apply(ctx as never, config as never)
    const reg = calls.registerAdapter as { providers: string[]; adapter: unknown }
    expect(reg.providers).toEqual([PROVIDER_ID])
    expect(typeof (reg.adapter as { stream: unknown }).stream).toBe('function')
    expect(calls.directory).toEqual([{
      provider: PROVIDER_ID,
      displayName: PROVIDER_NAME,
      settingsNs: 'claude-auth',
      settingsPath: [],
      declared: true,
    }])
    expect((calls.section as unknown[])[1]).toBe('claude-auth')
  })

  test('apply reports a diagnostic on the provider entry when the credentials file is missing', () => {
    const { ctx, calls } = makeCtx()
    const config = plugin.Config({ credentialsPath: join(dir, 'missing.json') })
    plugin.apply(ctx as never, config as never)
    const entries = calls.directory as Array<Record<string, unknown>>
    expect(entries).toHaveLength(1)
    expect(entries[0].provider).toBe(PROVIDER_ID)
    expect(entries[0].declared).toBe(true)
    expect(entries[0].error).toMatch(/No Claude Code OAuth credentials found/)
  })
})

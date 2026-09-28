import { afterEach, beforeEach, describe, expect, test } from 'bun:test'
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import * as plugin from '../src/index.ts'
import { PROVIDER_ID, PROVIDER_NAME } from '../src/adapter.ts'

// dsh 0.1.7 volatile config fields are cosmokit volatile refs: a stable value
// read with `.get()` and an owning-runtime write behind a global symbol. The
// 0.1.7 parser emits these refs, so the suite consumes the real `Config`
// output and mutates a field through the write symbol to simulate the loader
// committing a new value in place (which never re-runs `apply`).
const VOLATILE_WRITE = Symbol.for('cosmokit.volatile.write')
function mutateVolatile<T>(ref: { get(): T }, value: T): void {
  ;(ref as unknown as Record<symbol, (v: never) => void>)[VOLATILE_WRITE](value as never)
}

interface DirectoryHandle {
  replacedCount: number
  replace: (next: unknown[]) => void
  dispose: () => void
}

function makeCtx() {
  const listeners: Record<string, Array<() => void>> = {}
  const calls: Record<string, unknown> = {}
  const ctx = {
    logger: { info: () => undefined, warn: () => undefined, error: () => undefined },
    on: (event: string, handler: () => void) => {
      ;(listeners[event] ??= []).push(handler)
    },
    effect: (fn: () => () => void, label?: string) => {
      calls.effect = label
      fn
    },
    llm: {
      registerAdapter: (providers: string[], adapter: unknown) => {
        calls.registerAdapter = { providers, adapter }
        return () => undefined
      },
      registerConfigurableProviders: (entries: unknown[]) => {
        calls.directory = entries
        const handle: DirectoryHandle = {
          replacedCount: 0,
          replace: (next: unknown[]) => {
            handle.replacedCount += 1
            calls.replaced = next
            calls.directory = next
          },
          dispose: () => undefined,
        }
        calls.directoryHandle = handle
        return handle
      },
    },
  }
  return { ctx, calls, listeners }
}
function makeFixtureCreds(): { dir: string; credsPath: string } {
  const dir = mkdtempSync(join(tmpdir(), 'dsh-plugin-'))
  const credsPath = join(dir, '.credentials.json')
  writeFileSync(credsPath, JSON.stringify({
    claudeAiOauth: { accessToken: 'a', refreshToken: 'r', expiresAt: Date.now() + 3_600_000 },
  }))
  return { dir, credsPath }
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
  test('apply registers the claude-subscription provider and a directory entry under settingsNs claude-auth', () => {
    const { dir, credsPath } = makeFixtureCreds()
    const { ctx, calls } = makeCtx()
    const config = plugin.Config({ credentialsPath: credsPath })
    plugin.apply(ctx as never, config as never)
    rmSync(dir, { recursive: true, force: true })
    const reg = calls.registerAdapter as { providers: string[]; adapter: unknown }
    expect(reg.providers).toEqual([PROVIDER_ID])
    expect(typeof (reg.adapter as { stream: unknown }).stream).toBe('function')
    // A valid fixture produces one entry with no diagnostic; the plugin reads
    // the volatile field live during the first build.
    expect((calls.directoryHandle as DirectoryHandle).replacedCount).toBe(0)
    expect(calls.directory).toEqual([{
      provider: PROVIDER_ID,
      displayName: PROVIDER_NAME,
      settingsNs: 'claude-auth',
      settingsPath: [],
      declared: true,
    }])
  })

  test('apply reports a diagnostic on the provider entry when the credentials file is missing', () => {
    const { dir } = makeFixtureCreds()
    const { ctx, calls } = makeCtx()
    const config = plugin.Config({ credentialsPath: join(dir, 'missing.json') })
    plugin.apply(ctx as never, config as never)
    rmSync(dir, { recursive: true, force: true })
    const entries = calls.directory as Array<Record<string, unknown>>
    expect(entries).toHaveLength(1)
    expect(entries[0].provider).toBe(PROVIDER_ID)
    expect(entries[0].declared).toBe(true)
    expect(entries[0].error).toMatch(/No Claude Code OAuth credentials found/)
  })
})

describe('dsh 0.1.7 volatile settings model', () => {
  // 0.1.7 removed settings.installSection: config fields are volatile. The
  // loader commits new values into the same refs in place and emits
  // 'loader/volatile-update' on the fiber ctx — the plugin refreshes the
  // provider directory from that event and never re-runs `apply`.
  test('a volatile change + loader/volatile-update re-calls the directory handle replace', async () => {
    const { dir, credsPath } = makeFixtureCreds()
    const { ctx, calls, listeners } = makeCtx()
    const config = plugin.Config({
      credentialsPath: credsPath,
      disabledModels: ['some-model'],
    })
    plugin.apply(ctx as never, config as never)
    const volatileListeners = listeners['loader/volatile-update'] ?? []
    expect(volatileListeners).toHaveLength(1)

    // Simulate the live-update path: the loader writes the new value into the
    // same volatile ref (no re-apply) and dispatches the event to the fiber.
    mutateVolatile(config.streamIdleTimeoutMs, 605_000 as never)
    for (const fire of volatileListeners) fire()
    // refreshDirectory coalesces the replace into a queueMicrotask — yield so it
    // runs before asserting.
    await Promise.resolve()

    const handle = calls.directoryHandle as DirectoryHandle
    expect(handle.replacedCount).toBe(1)
    const replaced = calls.replaced as Array<{ provider: string; settingsNs: string; declared: boolean }>
    expect(replaced).toHaveLength(1)
    expect(replaced[0].provider).toBe(PROVIDER_ID)
    expect(replaced[0].settingsNs).toBe('claude-auth')
    expect(replaced[0].declared).toBe(true)
    // The refreshed read saw the same valid fixture, so no diagnostic surfaced.
    expect((replaced[0] as Record<string, unknown>).error).toBeUndefined()
    rmSync(dir, { recursive: true, force: true })
  })
})

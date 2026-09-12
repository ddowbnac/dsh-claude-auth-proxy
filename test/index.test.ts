import { describe, expect, test } from 'bun:test'
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

  test('apply registers the claude-subscription provider and a settings section', () => {
    const { ctx, calls } = makeCtx()
    const config = plugin.Config({})
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
})

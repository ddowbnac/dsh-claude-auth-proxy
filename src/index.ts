import z from '@deepseek-ai/schemastery'
import { homedir } from 'node:os'
import { isAbsolute, join } from 'node:path'
import type { Context } from '@deepseek-ai/cordis'
import {
  ClaudeSubscriptionAdapter,
  DEFAULT_REQUEST_TIMEOUT_MS,
  DEFAULT_STREAM_IDLE_TIMEOUT_MS,
  PROVIDER_ID,
  PROVIDER_NAME,
} from './adapter.ts'
import { ClaudeCredentialStore } from './auth.ts'
import { DEFAULT_CATALOG } from './model-config.ts'
import type { ClaudeConnectionOptions } from './types.ts'

export { ClaudeSubscriptionAdapter } from './adapter.ts'
export { ClaudeCredentialStore, unavailableCredential, missingEntryCredential, isUnavailableCredential } from './auth.ts'
export { DEFAULT_CATALOG } from './model-config.ts'
export { PROVIDER_ID, PROVIDER_NAME } from './adapter.ts'
export type { ClaudeCatalogModel, ClaudeConnectionOptions } from './types.ts'
export type { ResolvedCredential, UnavailableCredential } from './auth.ts'

export const name = 'claude-auth'
export const inject = ['llm'] as const

const NS = 'claude-auth'

export const Config = z.object({
  credentialsPath: z.string().default(''),
  disabledModels: z.array(z.string()).default([]),
  streamIdleTimeoutMs: z.number().step(1).min(5_000).default(DEFAULT_STREAM_IDLE_TIMEOUT_MS),
  requestTimeoutMs: z.number().step(1).min(30_000).default(DEFAULT_REQUEST_TIMEOUT_MS),
})

type ConfigValue = ReturnType<typeof Config>

function resolveConnection(config: ConfigValue): ClaudeConnectionOptions {
  return {
    credentialsPath: config.credentialsPath,
    disabledModels: config.disabledModels,
    streamIdleTimeoutMs: config.streamIdleTimeoutMs,
    requestTimeoutMs: config.requestTimeoutMs,
  }
}

function effectiveCredentialsPath(raw: string): string {
  const expanded = raw === '~' ? homedir() : raw.startsWith('~/') ? join(homedir(), raw.slice(2)) : raw
  return isAbsolute(expanded) ? expanded : join(homedir(), '.claude', '.credentials.json')
}

const CREDENTIAL_REF = `${PROVIDER_ID.toUpperCase().replace(/[^A-Z0-9]+/g, '_')}_API_KEY`

function deriveCredentialRef(provider: string): string {
  return `${provider.toUpperCase().replace(/[^A-Z0-9]+/g, '_')}_API_KEY`
}

export function apply(ctx: Context, config: ConfigValue): void {
  let current = (): ConfigValue => config
  let store: ClaudeCredentialStore | undefined
  let lastSyncedToken: string | undefined

  const credentialsService = () => {
    try {
      return ctx.get('credentials') as { set: (ref: unknown, value: string) => Promise<void> } | undefined
    } catch {
      return undefined
    }
  }

  const syncCredentialRef = async (): Promise<void> => {
    try {
      const { credentialRef } = await import('@deepseek-ai/dsh-credentials')
      const active = storeNow()
      const creds = active.read()
      if (creds === null || creds.expiresAt < Date.now()) {
        ctx.logger.warn(`claude-auth: credentials not valid, skipping credential ref sync (path=${active.credentialsPath})`)
        return
      }
      if (creds.accessToken === lastSyncedToken) return
      const service = credentialsService()
      if (service === undefined) {
        ctx.logger.warn('claude-auth: credentials service not available, skipping credential ref sync')
        return
      }
      const ref = credentialRef(CREDENTIAL_REF)
      await service.set(ref, creds.accessToken)
      lastSyncedToken = creds.accessToken
      ctx.logger.info(`claude-auth: mirrored access token to ${CREDENTIAL_REF} credential ref`)
    } catch (e) {
      ctx.logger.warn(`claude-auth: credential ref sync failed: ${e}`)
    }
  }

  const storeNow = (): ClaudeCredentialStore => {
    const conn = resolveConnection(current())
    const path = effectiveCredentialsPath(conn.credentialsPath)
    if (store === undefined || store.credentialsPath !== path) {
      store?.stop()
      store = new ClaudeCredentialStore(path)
      store.startProactiveRefresh()
      lastSyncedToken = undefined
    }
    return store
  }

  const adapter = new ClaudeSubscriptionAdapter({
    resolveConnection: () => resolveConnection(current()),
    resolveCredential: (signal) => storeNow().resolve(signal, 60_000),
    resolveModels: async (signal) => {
      const disabled = new Set(resolveConnection(current()).disabledModels)
      const discovered = await storeNow().discoverModels(signal).catch(() => undefined)
      const base = discovered ?? DEFAULT_CATALOG
      return base.filter(model => !disabled.has(model.id))
    },
    log: (message) => ctx.logger.info(`claude-auth: ${message}`),
    resolveAttachments: () => {
      try {
        return ctx.get('attachments') as import('./serialize.ts').AttachmentReader | undefined
      } catch {
        return undefined
      }
    },
  })

  ctx.llm.registerAdapter([PROVIDER_ID], adapter)

  const directoryHandle = typeof ctx.llm.registerConfigurableProviders === 'function'
    ? ctx.llm.registerConfigurableProviders([directoryEntry()])
    : undefined

  function directoryEntry() {
    const active = storeNow()
    const creds = active.read()
    const hasValidCreds = creds !== null && creds.expiresAt > Date.now()
    return {
      provider: PROVIDER_ID,
      displayName: PROVIDER_NAME,
      settingsNs: NS,
      settingsPath: [],
      declared: true,
      ...creds === null
        ? { error: `No Claude Code OAuth credentials found at ${active.credentialsPath} — run \`claude\` once to authenticate.` }
        : hasValidCreds ? {} : { error: `Claude Code OAuth credentials at ${active.credentialsPath} are expired — run \`claude\` to re-authenticate.` },
    }
  }

  let refreshDirectoryScheduled = false
  const refreshDirectory = () => {
    if (refreshDirectoryScheduled || directoryHandle === undefined) return
    refreshDirectoryScheduled = true
    queueMicrotask(() => {
      refreshDirectoryScheduled = false
      try {
        directoryHandle.replace([directoryEntry()])
      } catch {
      }
    })
  }

  ctx.inject(['settings'], (settingsCtx) => {
    (settingsCtx as any).settings.installSection(ctx, NS, Config, config, {
      setSource: (source: () => ConfigValue) => {
        current = source
      },
      onChange: () => {
        refreshDirectory()
      },
    })
  })

  storeNow()
  void syncCredentialRef().catch((e: unknown) => ctx.logger.warn(`claude-auth: initial credential sync failed: ${e}`))

  ctx.effect(() => () => {
    store?.stop()
    directoryHandle?.()
  }, 'claude-auth: dispose')
}

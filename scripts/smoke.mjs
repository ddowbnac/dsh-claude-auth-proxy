#!/usr/bin/env node
import { ClaudeSubscriptionAdapter } from '../src/adapter.ts'
import { ClaudeCredentialStore } from '../src/auth.ts'
import { DEFAULT_CATALOG } from '../src/model-config.ts'
import { PROVIDER_ID } from '../src/adapter.ts'

const model = process.argv[2] ?? 'claude-sonnet-5'

const store = new ClaudeCredentialStore(process.env.CLAUDE_CREDENTIALS_PATH || undefined)
const connection = {
  credentialsPath: process.env.CLAUDE_CREDENTIALS_PATH ?? '',
  disabledModels: [],
  streamIdleTimeoutMs: 120_000,
  requestTimeoutMs: 180_000,
}

const adapter = new ClaudeSubscriptionAdapter({
  resolveConnection: () => connection,
  resolveCredential: (signal) => store.resolve(signal),
  resolveModels: async () => [...DEFAULT_CATALOG],
})

const { createUserMessage } = await import('@deepseek-ai/dsh-llm')

const started = Date.now()
let text = ''
try {
  for await (const chunk of adapter.stream({
    provider: PROVIDER_ID,
    model,
    messages: [createUserMessage({ content: [{ type: 'text', text: 'Reply with the single word: pong' }] })],
    maxTokens: 200,
  })) {
    if (chunk.type === 'text-delta') {
      text += chunk.text
      process.stdout.write(chunk.text)
    } else if (chunk.type === 'usage') {
      console.error(`\n[smoke] usage: input=${chunk.usage.inputTokens} output=${chunk.usage.outputTokens}`)
    } else if (chunk.type === 'finish') {
      console.error(`\n[smoke] finish: ${JSON.stringify(chunk.reason)}`)
    }
  }
  console.error(`\n[smoke] OK in ${Date.now() - started}ms`)
} catch (error) {
  console.error(`\n[smoke] FAILED: ${error?.message ?? error}`)
  console.error(error?.cause ?? '')
  process.exit(1)
} finally {
  store.stop()
}

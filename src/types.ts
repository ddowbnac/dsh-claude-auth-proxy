export interface ClaudeCatalogModel {
  id: string
  name?: string
  description?: string
  contextWindow: number
  maxTokens: number
  reasoningEfforts?: string[]
  defaultReasoningEffort?: string
}

export interface ClaudeConnectionOptions {
  credentialsPath: string
  disabledModels: readonly string[]
  streamIdleTimeoutMs: number
  requestTimeoutMs: number
}

export interface ClaudeOAuthCredentials {
  accessToken: string
  refreshToken: string
  expiresAt: number
  refreshTokenExpiresAt?: number
  scopes?: string[]
  subscriptionType?: string
}

export const OAUTH_TOKEN_URL = 'https://claude.ai/v1/oauth/token'
export const OAUTH_CLIENT_ID = '9d1c250a-e61b-44d9-88ed-5944d1962f5e'

export const API_BASE_URL = 'https://api.anthropic.com'

export const SYSTEM_IDENTITY = 'You are Claude Code, Anthropic\'s official CLI for Claude.'

export const ANTHROPIC_API_VERSION = '2023-06-01'

export interface WireTextBlock {
  type: 'text'
  text: string
  cache_control?: { type: string }
}

export interface WireThinkingBlock {
  type: 'thinking'
  thinking: string
  signature?: string
}

export interface WireImageBlock {
  type: 'image'
  source: { type: 'base64'; media_type: string; data: string }
}

export interface WireToolUseBlock {
  type: 'tool_use'
  id: string
  name: string
  input: unknown
}

export interface WireToolResultBlock {
  type: 'tool_result'
  tool_use_id: string
  content: string | WireBlock[]
  is_error?: boolean
}

export type WireBlock = WireTextBlock | WireThinkingBlock | WireImageBlock | WireToolUseBlock | WireToolResultBlock

export interface WireMessage {
  role: 'system' | 'user' | 'assistant'
  content: string | WireBlock[]
}

export interface WireTool {
  name: string
  description?: string
  input_schema: Record<string, unknown>
}

export interface WireRequest {
  model: string
  messages: WireMessage[]
  system?: string | WireTextBlock[]
  tools?: WireTool[]
  max_tokens: number
  temperature?: number
  stop?: string[]
  stream: true
}

export interface WireSseEvent {
  type: string
  [key: string]: unknown
}

export interface WireUsage {
  input_tokens?: number
  output_tokens?: number
  cache_creation_input_tokens?: number
  cache_read_input_tokens?: number
}

export interface WireError {
  type?: string
  error?: { type?: string; message?: string }
  [key: string]: unknown
}

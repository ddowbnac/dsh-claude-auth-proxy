# @djdowbnac/dsh-claude-auth

dsh model provider that streams Claude through your Claude Code subscription. No Anthropic API key.

Registers a `claude-subscription` provider on `ctx.llm`. On the first request the plugin queries `GET /v1/models` with your OAuth token and serves every model your account can use, with its real context window and output cap. The catalog updates itself when Anthropic ships or retires models, so there is no static model list to maintain.

Requests go to `api.anthropic.com/v1/messages` with the Claude Code first-party headers. Credentials come from `~/.claude/.credentials.json`, the same file the `claude` CLI writes.

> Anthropic reserves subscription tokens for official clients. This is a community workaround, same class as `opencode-claude-auth`. It draws from your plan limits, not API billing, and may stop working when Anthropic changes its OAuth surface.

## How refresh works

The plugin keeps an access token in memory for 30 seconds and only calls the token endpoint when the token is within 60 seconds of expiry, or once per hour as a background pre-refresh. A cross-process lock serializes concurrent instances: the loser waits up to 15 seconds and adopts the winner's token instead of refreshing again. Anthropic rotates refresh tokens on single use, so double refreshes were what broke tokens before.

## Install

Prerequisites: dsh on the `0.1.5-rc` line, bun, and a one-time `claude` login.

```sh
dsh plugin --profile <name> add @djdowbnac/dsh-claude-auth
```

That is the whole install. The command registers the package in the profile's `dsh.profile.bundles`, and dsh mounts the plugin's `cordis.patch.yml` as an automatic bundle layer on every boot. No manual patch entry, no profile edits.

For a local checkout, `add .` from inside the checkout directory. Restart dsh and pick a `claude-subscription` model.

## Settings

Namespace `claude-auth`:

| Key | Default | Meaning |
|---|---|---|
| `credentialsPath` | `''` (→ `~/.claude/.credentials.json`) | Override the credentials file path. |
| `disabledModels` | `[]` | Model ids to stop serving. |
| `streamIdleTimeoutMs` | `600000` | Abort the stream when SSE goes quiet this long. |
| `requestTimeoutMs` | `900000` | Per-request budget. |

Changes apply on the next request. A missing or invalid credentials file shows as a diagnostic on the provider entry, and requests fail with a typed error.

## Web settings

The Models page in the dsh web UI shows a green credential dot next to the provider when your Claude subscription credentials are valid. The plugin mirrors the current access token into the dsh credential store under the `CLAUDE_SUBSCRIPTION_API_KEY` reference, which is the same reference the page checks. When the token expires or the credentials file disappears, the dot goes away and the provider row shows a diagnostic.

The provider editor card is read-only by design in the dsh web UI. The page renders curated field layouts only for the two first-party provider namespaces (`llm-deepseek`, `llm-pi-ai`); every other namespace falls back to the "edit settings.yaml directly" hint. To change `streamIdleTimeoutMs` or `disabledModels`, edit the `claude-auth` section in `~/.dsh/settings.yaml`.

Reasoning effort is not a provider setting. Pick it in the model selection popup: the effort selector appears for every model that supports adaptive thinking (all current Opus, Sonnet, Fable, and Haiku 4.5+ models). The chosen effort is sent as `output_config.effort` on the wire request.

## Develop

```sh
bun test        # 72 tests, all offline
bun run build
bun run smoke   # one live request, costs a few tokens
```

| File | What it does |
|---|---|
| `src/index.ts` | Plugin entry: config schema, provider registration, settings, disposal. |
| `src/adapter.ts` | `LlmAdapter`: headers, fetch, SSE consumption, error mapping, model discovery. |
| `src/serialize.ts` | Harness `Message[]` to Anthropic wire request. |
| `src/translate.ts` | Anthropic SSE to harness `StreamChunk`s. |
| `src/sse.ts` | SSE framing parser. |
| `src/auth.ts` | Credential read, refresh, write-back, backoff, cross-process lock. |
| `src/model-config.ts` | Beta headers, effort support, catalog fallback. |

Unit tests inject `fetchImpl`, so they run offline. The smoke script is the only live path.

## Limitations

- Text only. Image and file content blocks return `UNSUPPORTED_CONTENT`.
- One credentials file per provider route. Multi-account is not wired.
- If the models endpoint is unreachable, the plugin falls back to a static catalog of the current Claude 5 lineup.

## References

- [deepseek-ai/deepseek-harness](https://github.com/deepseek-ai/deepseek-harness): `dsh-llm` adapter contract, `dsh-settings`, `dsh-timeout`.
- [griffinmartin/opencode-claude-auth](https://github.com/griffinmartin/opencode-claude-auth): upstream reference for the OAuth surface and refresh strategy.
- [Claude model docs](https://platform.claude.com/docs/en/models/overview): current lineup and model ids.

## License

MIT

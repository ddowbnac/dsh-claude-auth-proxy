import { createHash } from 'node:crypto'

const BILLING_SALT = '59cf53e54c78'

export interface BillingInput {
  firstUserMessageText: string
  cliVersion: string
  entrypoint: string
}

export function buildBillingHeader({ firstUserMessageText, cliVersion, entrypoint }: BillingInput): string {
  const cch = createHash('sha256').update(firstUserMessageText).digest('hex').slice(0, 5)
  const sampled = [4, 7, 20].map(i => (i < firstUserMessageText.length ? firstUserMessageText[i] : '0')).join('')
  const suffix = createHash('sha256').update(`${BILLING_SALT}${sampled}${cliVersion}`).digest('hex').slice(0, 3)
  return `x-anthropic-billing-header: cc_version=${cliVersion}.${suffix}; cc_entrypoint=${entrypoint}; cch=${cch};`
}

export function stainlessHeaders(): Record<string, string> {
  return {
    'x-stainless-arch': process.arch === 'arm64' ? 'arm64' : process.arch,
    'x-stainless-lang': 'js',
    'x-stainless-os': process.platform === 'darwin' ? 'MacOS' : process.platform,
    'x-stainless-package-version': '0.81.0',
    'x-stainless-retry-count': '0',
    'x-stainless-runtime': 'node',
  }
}

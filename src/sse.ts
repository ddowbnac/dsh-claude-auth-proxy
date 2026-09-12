import type { WireSseEvent } from './types.ts'

const decode = new TextDecoder('utf-8')

export interface SseParser {
  push(chunk: Uint8Array): WireSseEvent[]
  end(): WireSseEvent[]
}

export function createSseParser(): SseParser {
  let buffer = ''

  function extractEvents(text: string): { events: WireSseEvent[]; remainder: string } {
    const events: WireSseEvent[] = []
    let index: number
    while ((index = text.indexOf('\n\n')) !== -1 || (index = text.indexOf('\r\n\r\n')) !== -1) {
      const frame = text.slice(0, index)
      const advance = frame.endsWith('\r') ? index + 4 : index + 2
      text = text.slice(advance)
      const data = parseFrame(frame)
      if (data !== undefined) events.push(data)
    }
    return { events, remainder: text }
  }

  function parseFrame(frame: string): WireSseEvent | undefined {
    const dataLines: string[] = []
    for (const rawLine of frame.split(/\r?\n/)) {
      if (rawLine.startsWith(':')) continue
      const colon = rawLine.indexOf(':')
      const value = colon === -1 ? rawLine : rawLine.slice(colon + 1).replace(/^ /, '')
      if (colon === -1 || rawLine.startsWith('data')) dataLines.push(value)
    }
    if (dataLines.length === 0) return undefined
    const payload = dataLines.join('\n')
    if (payload === '' || payload === '[DONE]') return undefined
    try {
      return JSON.parse(payload) as WireSseEvent
    } catch {
      return undefined
    }
  }

  return {
    push(chunk) {
      buffer += decode.decode(chunk, { stream: true })
      const { events, remainder } = extractEvents(buffer)
      buffer = remainder
      return events
    },
    end() {
      buffer += decode.decode()
      const { events, remainder } = extractEvents(buffer)
      buffer = remainder
      if (buffer.trim().length > 0) {
        const trailing = parseFrame(buffer.replace(/\r?\n$/, ''))
        if (trailing !== undefined) events.push(trailing)
        buffer = ''
      }
      return events
    },
  }
}

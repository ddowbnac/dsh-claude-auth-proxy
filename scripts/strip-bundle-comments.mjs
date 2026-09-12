#!/usr/bin/env node
import { readFileSync, writeFileSync } from 'node:fs'
const [, , file, ...rest] = process.argv
if (!file || !rest.includes('--write')) {
  console.error('usage: node scripts/strip-bundle-comments.mjs <file> --write')
  process.exit(1)
}
let text = readFileSync(file, 'utf8')
const lines = text.split('\n')
let i = 0
while (i < lines.length && (lines[i].startsWith('//') || lines[i].trim() === '')) i++
if (i > 0 && i < lines.length) {
  text = lines.slice(i).join('\n')
  writeFileSync(file, text)
  console.log(`stripped ${i} leading comment/blank lines from ${file}`)
}

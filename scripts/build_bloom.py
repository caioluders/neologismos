#!/usr/bin/env python3
import re, math, base64, sys

VALID = re.compile(r'^[a-zàáâãäéêëíïóôõöúüçñ]+$')
FREQ_MIN = 10

# 1) dicionário base (lemas do pt-br)
words = set()
with open('palavras.txt', encoding='utf-8') as f:
    for line in f:
        w = line.strip().lower()
        if len(w) >= 2 and VALID.match(w):
            words.add(w)
base_n = len(words)

# 2) formas flexionadas/gramaticais reais (lista de frequência OpenSubtitles pt-br)
with open('freq.txt', encoding='utf-8') as f:
    for line in f:
        p = line.split()
        if len(p) != 2:
            continue
        w = p[0].lower()
        try:
            c = int(p[1])
        except ValueError:
            continue
        if c >= FREQ_MIN and len(w) >= 2 and VALID.match(w):
            words.add(w)

n = len(words)
bits_per = 8                                   # ~2% de falso positivo
m = n * bits_per
k = max(1, round(bits_per * math.log(2)))
print(f'base: {base_n}, uniao: {n}, bits: {m} ({m/8/1024:.0f} KB), k: {k}', file=sys.stderr)

MASK = 0xFFFFFFFF
def fnv1a(bs, h):
    for b in bs:
        h ^= b
        h = (h * 16777619) & MASK
    return h

bits = bytearray((m + 7) // 8)
for w in words:
    bs = w.encode('utf-8')
    h1 = fnv1a(bs, 2166136261)
    h2 = fnv1a(bs, 1099511628211 & MASK) | 1
    for i in range(k):
        idx = (h1 + i * h2) % m
        bits[idx >> 3] |= (1 << (idx & 7))

b64 = base64.b64encode(bytes(bits)).decode('ascii')
with open('words.js', 'w', encoding='utf-8') as out:
    out.write(f'window.WORDS={{m:{m},k:{k},bits:"{b64}"}};\n')

import os
print(f'words.js: {os.path.getsize("words.js")/1024:.0f} KB', file=sys.stderr)

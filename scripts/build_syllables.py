#!/usr/bin/env python3
import re, json, collections, sys
import pyphen

dic = pyphen.Pyphen(lang='pt_BR')

# aceita só palavras com letras do português (inclui acentos), minúsculas, sem hífen/ponto
VALID = re.compile(r'^[a-zàáâãäéêëíïóôõöúüçñ]+$')

ini = collections.Counter()   # sílaba inicial
mid = collections.Counter()   # sílaba do meio
fim = collections.Counter()   # sílaba final
mono = collections.Counter()  # palavra de 1 sílaba

n_words = 0
with open('palavras.txt', encoding='utf-8') as f:
    for line in f:
        w = line.strip().lower()
        if len(w) < 3 or not VALID.match(w):
            continue
        sylls = dic.inserted(w).split('-')
        sylls = [s for s in sylls if s]
        if not sylls:
            continue
        n_words += 1
        if len(sylls) == 1:
            mono[sylls[0]] += 1
            continue
        ini[sylls[0]] += 1
        fim[sylls[-1]] += 1
        for s in sylls[1:-1]:
            mid[s] += 1

def top(counter, n, minc=2):
    items = [(s, c) for s, c in counter.items() if c >= minc]
    items.sort(key=lambda x: -x[1])
    return items[:n]

data = {
    'ini': top(ini, 2500),
    'mid': top(mid, 2500),
    'fim': top(fim, 2500),
}

print(f'palavras usadas: {n_words}', file=sys.stderr)
for k, v in data.items():
    print(f'{k}: {len(v)} sílabas únicas, ex: {[s for s,_ in v[:12]]}', file=sys.stderr)

with open('data.js', 'w', encoding='utf-8') as out:
    out.write('window.SYLLABLES=')
    json.dump(data, out, ensure_ascii=False, separators=(',', ':'))
    out.write(';\n')

import os
print(f'data.js: {os.path.getsize("data.js")/1024:.1f} KB', file=sys.stderr)

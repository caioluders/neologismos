#!/usr/bin/env python3
# Gera palavras do model.json (forward em numpy puro, sem tinygrad/GPU).
import json, sys, re
import numpy as np

MODEL = sys.argv[3] if len(sys.argv) > 3 else 'model.json'
m = json.load(open(MODEL))
itos = m['vocab']; V = len(itos); stoi = {c: i for i, c in enumerate(itos)}
BLOCK, E, H = m['block'], m['n_emb'], m['n_hid']
C  = np.array(m['C']).reshape(V, E)
W1 = np.array(m['W1']).reshape(H, BLOCK * E); b1 = np.array(m['b1'])
W2 = np.array(m['W2']).reshape(V, H);         b2 = np.array(m['b2'])

# conjunto de palavras reais (pra marcar as que já existem)
VALID = re.compile(r'^[a-zàáâãéêíóôõúüç]+$')
reais = {w.strip().lower() for w in open('palavras.txt', encoding='utf-8')
         if len(w.strip()) >= 3 and VALID.match(w.strip().lower())}

def gera(temp=0.9):
    ctx = [0] * BLOCK; out = []
    for _ in range(30):
        e = C[ctx].reshape(-1)
        h = np.tanh(e @ W1.T + b1)
        logits = (h @ W2.T + b2) / temp
        p = np.exp(logits - logits.max()); p /= p.sum()
        ix = np.random.choice(V, p=p)
        if ix == 0: break
        out.append(itos[ix]); ctx = ctx[1:] + [ix]
    return ''.join(out)

n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
temp = float(sys.argv[2]) if len(sys.argv) > 2 else 0.9
np.random.seed()
for _ in range(n):
    w = gera(temp)
    print(f'  {w}' + ('  (já existe)' if w in reais else ''))

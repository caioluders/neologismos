#!/usr/bin/env python3
# Mini-LLM de 1 palavra: rede neural char-level estilo "makemore" (Bengio 2003).
# Treina na GPU AMD via tinygrad. Exporta pesos pra rodar no navegador (JS).
import re, json, random, time, sys
import numpy as np
from tinygrad import Tensor, TinyJit, Context
from tinygrad.nn import Embedding, Linear
from tinygrad.nn.optim import Adam
from tinygrad.nn.state import get_parameters

# ---------------- hiperparâmetros ----------------
BLOCK = 5       # quantas letras de contexto o modelo enxerga
N_EMB = 24      # tamanho do vetor (embedding) de cada letra
N_HID = 128     # neurônios da camada escondida
BATCH = 4096
STEPS = 6000
LR = 1e-3
SEED = 42
random.seed(SEED); np.random.seed(SEED); Tensor.manual_seed(SEED)

# ---------------- 1) dados: palavras -> caracteres ----------------
VALID = re.compile(r'^[a-zàáâãéêíóôõúüç]+$')  # letras do pt-br (sem 'ä/ë/ï/ö/ñ' raríssimos)
words = []
for line in open('palavras.txt', encoding='utf-8'):
    w = line.strip().lower()
    if len(w) >= 3 and VALID.match(w):      # sem teto de tamanho
        words.append(w)
random.shuffle(words)
print(f'palavras: {len(words)}', file=sys.stderr)

# vocabulário: '.' (fronteira/início/fim) + letras ordenadas
chars = sorted(set(''.join(words)))
itos = ['.'] + chars               # index -> caractere
stoi = {c: i for i, c in enumerate(itos)}
V = len(itos)
print(f'vocab ({V}): {"".join(itos)}', file=sys.stderr)

# ---------------- 2) dataset (contexto -> próxima letra) ----------------
# Para cada palavra "casa": contexto desliza e o alvo é a próxima letra.
# "....." -> c | "....c" -> a | "...ca" -> s | "..cas" -> a | ".casa" -> . (fim)
Xs, Ys = [], []
for w in words:
    ctx = [0] * BLOCK                      # começa tudo em '.'
    for ch in w + '.':                     # '.' no fim = sinal de "acabou a palavra"
        ix = stoi[ch]
        Xs.append(ctx); Ys.append(ix)
        ctx = ctx[1:] + [ix]               # desliza a janela
X = np.array(Xs, dtype=np.int32)
Y = np.array(Ys, dtype=np.int32)
print(f'exemplos de treino: {len(X)}', file=sys.stderr)

# ---------------- 3) modelo ----------------
class Modelo:
    def __init__(self):
        self.C  = Embedding(V, N_EMB)          # tabela de embeddings das letras
        self.l1 = Linear(BLOCK * N_EMB, N_HID) # camada escondida
        self.l2 = Linear(N_HID, V)             # saída: 1 score por letra possível
    def __call__(self, x):                     # x: (B, BLOCK) inteiros
        e = self.C(x)                          # (B, BLOCK, N_EMB)
        e = e.reshape(e.shape[0], -1)          # (B, BLOCK*N_EMB) -> concatena as letras
        h = (self.l1(e)).tanh()                # (B, N_HID)
        return self.l2(h)                      # (B, V) logits

model = Modelo()
params = get_parameters(model)
opt = Adam(params, lr=LR)
n_params = sum(int(np.prod(p.shape)) for p in params)
print(f'parâmetros: {n_params}', file=sys.stderr)

# ---------------- 4) treino (descida de gradiente) ----------------
Xt, Yt = Tensor(X), Tensor(Y)   # ficam na GPU

@TinyJit
def step(idx):
    xb, yb = Xt[idx], Yt[idx]
    logits = model(xb)
    # cross-entropy = -log P(letra certa). Minimizar = acertar mais o próximo caractere.
    loss = logits.sparse_categorical_crossentropy(yb)
    opt.zero_grad(); loss.backward(); opt.step()
    return loss

t0 = time.time()
with Context(TRAINING=1):            # habilita TRAINING pro otimizador
    for s in range(STEPS):
        idx = Tensor(np.random.randint(0, len(X), BATCH).astype(np.int32))
        loss = step(idx)
        if s % 500 == 0 or s == STEPS - 1:
            print(f'step {s:5d} | loss {float(loss.numpy()):.3f} | {time.time()-t0:.1f}s', file=sys.stderr)
print(f'treino: {time.time()-t0:.1f}s', file=sys.stderr)

# ---------------- 5) amostragem (gerar palavras) ----------------
Tensor.training = False
W1 = model.l1.weight.numpy(); B1 = model.l1.bias.numpy()
W2 = model.l2.weight.numpy(); B2 = model.l2.bias.numpy()
Cw = model.C.weight.numpy()

def gera(temp=0.9):
    ctx = [0] * BLOCK
    out = []
    for _ in range(30):
        e = Cw[ctx].reshape(-1)
        h = np.tanh(e @ W1.T + B1)
        logits = (h @ W2.T + B2) / temp
        p = np.exp(logits - logits.max()); p /= p.sum()
        ix = np.random.choice(V, p=p)
        if ix == 0: break
        out.append(itos[ix]); ctx = ctx[1:] + [ix]
    return ''.join(out)

realset = set(words)
print('\n=== amostras (temp=0.9) ===', file=sys.stderr)
novas = 0
for _ in range(40):
    w = gera(0.9)
    tag = '' if (w and w not in realset) else '  <existe/vazia>'
    if w and w not in realset: novas += 1
    print(f'  {w}{tag}', file=sys.stderr)
print(f'\n{novas}/40 inéditas', file=sys.stderr)

# ---------------- 6) exporta pesos pro navegador ----------------
def q(a): return [round(float(x), 5) for x in np.array(a).reshape(-1)]
out = {
    'block': BLOCK, 'n_emb': N_EMB, 'n_hid': N_HID, 'vocab': itos,
    'C': q(Cw), 'W1': q(W1), 'b1': q(B1), 'W2': q(W2), 'b2': q(B2),
}
json.dump(out, open('model.json', 'w'), separators=(',', ':'))
import os
print(f'\nmodel.json: {os.path.getsize("model.json")/1024:.0f} KB', file=sys.stderr)

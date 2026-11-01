# NeologIA OS 98

Um gerador de **palavras que ainda não existem** em português — com cara de **Windows 98** e uma **mini-rede neural** rodando dentro do navegador.

**▶ Ao vivo: https://lude.rs/neologismos/**

Tudo é **100% client-side** (estático, sem backend): dá pra hospedar no GitHub Pages abrindo só o `index.html`.

![captura](https://img.shields.io/badge/Windows-98-008080?style=flat-square) ![client-side](https://img.shields.io/badge/100%25-client--side-000080?style=flat-square) ![tinygrad](https://img.shields.io/badge/treinado%20com-tinygrad-black?style=flat-square)

---

## O que é

Dois "motores" inventam palavras plausíveis que **não constam no português**:

### neural (padrão)
Uma **mini-LLM a nível de caractere** (um MLP no estilo *makemore* do Karpathy — o avô conceitual do GPT). Ela gera a palavra **letra por letra**, prevendo a próxima com base nas anteriores. Foi **treinada com [tinygrad](https://github.com/tinygrad/tinygrad)** em ~268 mil palavras do português e exportada pra `model.json` (~21 mil parâmetros, ~180 KB). O *forward pass* é reimplementado em **JavaScript puro**, então a rede roda no seu navegador sem framework nenhum.

O painel **"a computação"** mostra, a cada letra, as probabilidades que a rede calculou — com a letra escolhida em destaque. A **temperatura** (0 a 1.5) controla o trade-off coerência × criatividade: baixa = conservador/determinístico, alta = ousado/caótico.

### sílabas
Sorteia sílabas reais (separadas com [pyphen](https://pyphen.org/) e **ponderadas por frequência**) e as emenda. Dá pra escolher nº de sílabas e o tipo (substantivo/adjetivo/verbo, forçando a terminação).

### Filtros (valem pros dois motores)
1. **Já existe?** — compara contra um **Bloom filter** de ~433 mil palavras reais (dicionário + formas flexionadas do OpenSubtitles). Se existe, re-sorteia.
2. **É pronunciável?** — descarta emendas impossíveis (3+ consoantes seguidas, letras/vogais repetidas) e re-sorteia.

### Interface Windows 98
Feita com [98.css](https://jdan.github.io/98.css/): wallpaper vintage, janela central, janela de histórico, **taskbar com relógio** e **menu Iniciar funcional** (gerar, histórico, como funciona, código-fonte, e um "Desligar..." de brincadeira). Ainda tem favoritos (salvos no navegador), links compartilháveis (`#palavra` na URL) e animação *slot-machine*.

---

## Estrutura

```
index.html   — a aplicação (HTML + CSS + JS, tudo inline)
data.js      — pool de sílabas por posição, ponderado por frequência (~75 KB)
words.js     — Bloom filter de ~433 mil palavras reais (~565 KB)
model.js     — pesos da mini-LLM neural (window.NEURAL, ~180 KB)
98.css       — tema Windows 98 (+ fontes ms_sans_serif.*)
scripts/     — pipeline offline que gera os arquivos acima
```

## Rodar localmente

É estático — basta abrir o `index.html` no navegador (os dados vêm como `<script>`, então funciona até via `file://`). Pra servir:

```bash
python3 -m http.server 8000   # http://localhost:8000
```

---

## Como os dados foram gerados (`scripts/`)

Lista de palavras base: [pythonprobr/palavras](https://github.com/pythonprobr/palavras) (`palavras.txt`). Lista de frequência: [HermitDave/FrequencyWords](https://github.com/hermitdave/FrequencyWords) (pt_br).

```bash
pip install pyphen numpy tinygrad

python scripts/build_syllables.py   # -> data.js   (sílabas por pyphen)
python scripts/build_bloom.py       # -> words.js  (Bloom filter)
python scripts/train.py             # -> model.json (treina a mini-LLM)
python scripts/sample.py 20 0.9     # amostra palavras de um model.json
```

### A mini-LLM em detalhe
Arquitetura (`scripts/train.py`): `Embedding(vocab, 24)` → concat de 5 letras de contexto → `Linear(120, 128)` + `tanh` → `Linear(128, 40)` → softmax. Treinada com cross-entropy + Adam. ~21 mil parâmetros, perda final ~2.05 (perplexidade ~8).

### Nota sobre a GPU
O treino roda em **CPU** via tinygrad. Tentamos a GPU (AMD RX 9070 XT / RDNA4, `gfx1201`) mas o backend nativo do tinygrad dá **HW fault** no *graph replay* do JIT nessa arquitetura nova (ver [tinygrad#17920](https://github.com/tinygrad/tinygrad) e a instabilidade geral de RDNA4 em compute). Com `JIT=0` a GPU não trava, mas fica lenta demais. Como o modelo é minúsculo, a CPU resolve em ~6 min. Os scripts `scripts/gpu.sh` / `scripts/cpu.sh` guardam a configuração de ambiente usada.

---

## Créditos
- [tinygrad](https://github.com/tinygrad/tinygrad) — treino da rede
- [98.css](https://jdan.github.io/98.css/) — tema Windows 98
- [pyphen](https://pyphen.org/) — separação silábica
- listas de palavras: [pythonprobr/palavras](https://github.com/pythonprobr/palavras), [HermitDave/FrequencyWords](https://github.com/hermitdave/FrequencyWords)
- ideia do modelo char-level: [makemore](https://github.com/karpathy/makemore) (Karpathy)

Feito por brincadeira. Use as palavras à vontade — elas não existem (ainda).

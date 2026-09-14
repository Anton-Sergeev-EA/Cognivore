# Cognivore

**Un framework per agenti multimodali local-first, senza PyTorch.**
Ragionamento LLM + RAG, con un indice vettoriale scritto a mano in C++ e
un'interfaccia web di chat -- tutto funziona su un normale laptop con sola
CPU, senza che nulla debba uscire dalla vostra macchina.

[![CI](https://github.com/Anton-Sergeev-EA/cognivore/actions/workflows/ci.yml/badge.svg)](https://github.com/Anton-Sergeev-EA/cognivore/actions/workflows/ci.yml)
[![CodeQL](https://github.com/Anton-Sergeev-EA/cognivore/actions/workflows/codeql.yml/badge.svg)](https://github.com/Anton-Sergeev-EA/cognivore/actions/workflows/codeql.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](../../LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](../../pyproject.toml)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-black)](https://github.com/astral-sh/ruff)

🌐 **Leggi questo in un'altra lingua:** [English](../../README.md) | [Русский](README.ru.md) | [Deutsch](README.de.md) | [Français](README.fr.md) | [Italiano](README.it.md) | [Español](README.es.md) | [中文](README.zh.md) | [日本語](README.ja.md) | [हिन्दी](README.hi.md)

Cognivore è un framework per agenti in stile ReAct con retrieval-augmented
generation e supporto per strumenti multimodali (trascrizione audio,
analisi delle scene video), progettato specificamente per funzionare
interamente su una macchina con sola CPU, senza alcuna dipendenza da
PyTorch in tutto lo stack. Il suo indice di retrieval è un core scritto a
mano in C++ (SIMD AVX2, ricerca esatta parallelizzata con OpenMP e un
grafo NSW approssimato scritto da zero), esposto a Python tramite
pybind11, con un fallback in puro NumPy in modo che `pip install` non
fallisca mai per la mancanza di un compilatore C++.

```
you> What does the ingested spec say about the retry policy, and what's 15% of that timeout in ms?

  [tool] search_knowledge_base({"query": "retry policy timeout"})
  [tool] calculator({"expression": "5000 * 0.15"})

cognivore> The spec (spec.md) sets a 5000ms timeout with exponential
backoff. 15% of that is 750ms.
```

Lo stesso ciclo, nella web UI -- una traccia dal vivo della chiamata allo strumento e del passaggio recuperato, in qualsiasi delle 9 lingue e dei 3 temi:

| Tema aurora, inglese -- traccia RAG | Tema scuro, russo -- calcolatrice |
|---|---|
| ![Web UI di Cognivore: tema aurora, inglese, una chiamata allo strumento search_knowledge_base e il passaggio recuperato](../screenshots/web-ui-en.png) | ![Web UI di Cognivore: tema scuro, russo, una chiamata allo strumento calculator](../screenshots/web-ui-ru.png) |

## Perché questo progetto esiste

La maggior parte dei progetti portfolio di "agenti AI" è un thin wrapper
attorno a una chiamata API. Questo progetto è costruito per mostrare
l'opposto: un framework in cui le parti interessanti (una vera struttura
dati ANN, un protocollo di tool-calling indipendente dal provider, una
pipeline di retrieval ibrida, una degradazione controllata quando le
dipendenze opzionali non sono presenti) sono implementate, non
semplicemente importate.

## Funzionalità

- **Ciclo dell'agente ReAct** (Thought → Action → Observation) che
  funziona con *qualsiasi* modello locale instruction-tuned, non solo con
  quelli sottoposti a fine-tuning per un formato specifico di function
  calling -- vedi
  [docs/architecture.md](../architecture.md#why-a-react-loop-instead-of-native-function-calling).
- **Indice vettoriale nativo in C++** (`native/vector_index.cpp`): un
  `FlatIndex` esatto (prodotto scalare AVX2/FMA, scansione parallela via
  OpenMP) e un `NSWIndex` approssimato (un grafo Navigable Small World a
  singolo livello scritto da zero -- inserimento, potatura dei vicini,
  ricerca greedy a fascio), entrambi collegati a Python tramite
  un'estensione pybind11 scritta a mano con uno stub di tipo `.pyi`. Passa
  automaticamente a un indice in puro NumPy se al momento
  dell'installazione non è presente un compilatore C++.
- **RAG ibrido**: chunking tramite splitter ricorsivo, embedding semantici
  `fastembed` (ONNX, senza PyTorch) con fallback a hashing-trick senza
  alcun download, retrieval ibrido vettore + BM25.
- **Strumenti multimodali**: calcolatrice sicura (basata su AST, senza
  `eval`), ricerca nella base di conoscenza, trascrizione audio con
  separazione approssimativa per parlante (`faster-whisper`) e
  rilevamento delle scene video (OpenCV) + OCR del testo a schermo
  (Tesseract) -- tutto solo su CPU, senza PyTorch.
- **Backend LLM collegabile**: inferenza locale GGUF tramite
  `llama-cpp-python`, oppure un `FakeLLMBackend` deterministico e senza
  dipendenze che percorre esattamente lo stesso codice di tool-calling
  senza alcun download -- quello su cui girano la test suite e la CI.
- **Interfaccia web**: FastAPI + streaming SSE + un'interfaccia di chat in
  JS/HTML/CSS puro (nessuno step di build, nessun framework) con
  drag-and-drop di file/audio/video, indicatori live di
  "pensiero"/connessione, temi scuro/chiaro/aurora e localizzazione in
  inglese, russo, tedesco, francese, italiano, spagnolo, cinese
  semplificato, giapponese e hindi.
- Suite di test completa, lint+formattazione con ruff, mypy (in modalità
  quasi strict, incluso uno stub per l'estensione nativa) e una matrice CI
  multi-OS/multi-Python.

## Avvio rapido

```bash
git clone https://github.com/Anton-Sergeev-EA/cognivore.git
cd cognivore
python3 -m venv .venv && source .venv/bin/activate
pip install -e .                 # compila l'estensione nativa se è presente un compilatore C++17

cognivore chat                   # chat interattiva, modalità demo offline per default
cognivore ingest ./docs          # indicizza una cartella di file markdown/testo
cognivore serve                  # web chat su http://127.0.0.1:8420
```

Per default non c'è alcun download di LLM e nessun accesso alla rete:
l'agente funziona con `FakeLLMBackend`, un piccolo backend deterministico
che comunque percorre il vero ciclo di tool-calling (provate `What is
12 * 7?` oppure `search the knowledge base for ...` dopo aver fatto
`ingest` di qualcosa). Per usare una vera LLM locale:

```bash
pip install -e ".[llm]"
# scaricate ad es. https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF
cp .env.example .env
echo 'COGNIVORE_LLM_MODEL_PATH=/path/to/model.gguf' >> .env
cognivore chat
```

Oppure, più semplicemente e senza alcun toolchain C++ coinvolto, puntate
invece a [Ollama](https://ollama.com): `COGNIVORE_LLM_PROVIDER=auto` (il
valore di default) prova prima un server Ollama in esecuzione
localmente, per poi ricadere su un percorso GGUF o su `FakeLLMBackend`:

```bash
ollama pull qwen2.5:3b   # funziona con qualsiasi modello instruction-tuned
cognivore chat           # rileva automaticamente il server Ollama in esecuzione
```

Gli strumenti audio/video richiedono i propri extra:
`pip install -e ".[audio,video]"` (oppure `.[all]` per tutto, GGUF
incluso). Consultate `.env.example` per tutte le impostazioni.

L'estrazione del testo a schermo nello strumento video (`analyze_video`)
richiede anche il *binario* OCR [Tesseract](https://github.com/tesseract-ocr/tesseract) --
il package `pytesseract` incluso dall'extra `video` è solo un sottile
wrapper attorno ad esso, e senza di esso l'OCR restituisce silenziosamente
nessun testo (il rilevamento delle scene e i timestamp continuano invece
a funzionare in ogni caso, poiché quella parte è puro OpenCV). Di default
riconosce inglese *e* russo (`eng+rus`, vedi `COGNIVORE_OCR_LANGUAGES` in
`.env.example`) -- su Debian/Ubuntu il semplice package `tesseract-ocr`
include automaticamente `eng` ma non `rus`, quindi installate entrambi
esplicitamente:

```bash
sudo apt install tesseract-ocr tesseract-ocr-rus   # Debian/Ubuntu
brew install tesseract                              # macOS -- include tutte le lingue insieme
# Windows: https://github.com/UB-Mannheim/tesseract/wiki (selezionate il russo nell'elenco delle lingue dell'installer)
```

L'immagine Docker le include già entrambe -- niente da installare in quel caso.

Richiedere una lingua il cui package di dati addestrati non è installato
non genera un errore -- riconosce silenziosamente in modo errato quella
scrittura come lettere latine simili (il cirillico "Контейнеры" diventa
"KoHTewHepbi"), il che sembra una scansione mal riuscita piuttosto che
una lingua mancante. Se il vostro testo a schermo usa un'altra lingua,
installate il suo package `tesseract-ocr-<lang>` e aggiungetelo a
`COGNIVORE_OCR_LANGUAGES` (per esempio `eng+rus+deu`).

### Docker

L'immagine è una build multi-stage (compila l'estensione nativa in C++,
poi scarta il compilatore per un'immagine runtime leggera), viene
eseguita come utente non-root e include un `HEALTHCHECK` -- costruita
una sola volta grazie al layer caching Docker di GitHub Actions e
verificata end-to-end (il server risponde effettivamente a
`/api/health`, il container riporta `healthy`, viene eseguito come
non-root) a ogni push, quindi non si limita a "compilare": viene
verificata.

**Il percorso più semplice -- niente Python, niente installazione di
Ollama, funziona allo stesso modo su Windows, macOS e Linux:**

```bash
docker compose up -d --build
docker compose exec ollama ollama pull qwen2.5:3b   # una tantum, ~2GB
```

Poi aprite <http://127.0.0.1:8420>. `docker-compose.yml` esegue anche
Ollama *dentro* il proprio container, quindi non c'è nulla da installare
sull'host oltre a Docker stesso; Cognivore lo raggiunge sulla rete di
compose tramite il nome del servizio (`http://ollama:11434`), il che
evita del tutto le solite differenze di networking dell'host tra
Windows/macOS/Linux. Senza quel `ollama pull` una tantum, Cognivore parte
e funziona comunque correttamente -- semplicemente ricade sulla modalità
demo offline `FakeLLMBackend` finché un modello non è disponibile.

Lo stack compose imposta anche `COGNIVORE_SEED_DEMO_KB=true`, quindi una base
di conoscenza nuova viene popolata automaticamente con due manuali aziendali
demo forniti (inglese + russo -- tariffe, SLA, sicurezza, politica di
rimborso, FAQ di supporto) invece di aprirsi su una dropzone vuota. Popola
sempre e solo una base di conoscenza *vuota*: una volta importati i propri
documenti, questo diventa un no-op permanente. Impostatelo su `false` in
`docker-compose.yml` (oppure `docker run -e COGNIVORE_SEED_DEMO_KB=false`) per
partire vuoti, o eseguite `cognivore seed-demo` in qualsiasi momento per
aggiungere gli stessi documenti demo a uno store già esistente.

**Avete già Ollama in esecuzione sull'host, oppure volete un unico
container?**

```bash
docker build -t cognivore .
docker run -d -p 8420:8420 -v cognivore-data:/data \
  -e COGNIVORE_OLLAMA_HOST=http://host.docker.internal:11434 \
  --add-host=host.docker.internal:host-gateway \
  cognivore
```

`host.docker.internal` viene fornito automaticamente su Docker Desktop
(Windows/macOS); l'`--add-host` esplicito riportato sopra è ciò che
permette allo stesso comando di funzionare anche su Linux "puro", dove
altrimenti questo nome non si risolverebbe.

L'immagine include gli strumenti audio/video (`faster-whisper`, OpenCV,
e Tesseract per l'OCR del testo a schermo) ma *non* `llama-cpp-python` -- comunica con Ollama tramite HTTP semplice
per l'LLM invece di caricare un file GGUF in-process, deliberatamente,
poiché llama-cpp-python non ha una wheel precompilata per ogni
piattaforma e richiede un compilatore che lo stage di runtime non porta
con sé. Volete comunque l'inferenza GGUF in-process dentro il container?
Aggiungete `build-essential` allo stage finale e riportate il suo
`pip install` all'extra `[all]`.

`faster-whisper` scarica il proprio modello di riconoscimento vocale da
Hugging Face la prima volta che la trascrizione audio viene effettivamente
usata (non al momento della build), proprio come `ollama pull` per l'LLM
-- con la differenza che qui il download avviene automaticamente al primo
utilizzo, senza bisogno di un comando esplicito. Come i modelli di Ollama,
viene memorizzato nella cache del volume persistente `cognivore-data`
(`HF_HOME=/data/hf-cache`), quindi si scarica una sola volta e non ad ogni
`docker compose up --build`.

**Immagine precompilata (senza alcuno step di build):** le release
taggate sono pubblicate multi-architettura (amd64 + arm64 -- inclusi
Apple Silicon e Raspberry Pi) su GHCR da
[`.github/workflows/docker-publish.yml`](../../.github/workflows/docker-publish.yml):

```bash
docker pull ghcr.io/anton-sergeev-ea/cognivore:latest
```

## Utilizzo

**CLI** (`cognivore --help` per l'elenco completo):

```bash
cognivore chat                              # REPL interattivo
cognivore ingest ./docs                     # indicizza ricorsivamente .txt/.md/.markdown/.rst
cognivore seed-demo                         # aggiunge i manuali aziendali demo EN+RU forniti
cognivore serve --host 0.0.0.0 --port 8420  # interfaccia web + API REST/SSE
cognivore bench                             # benchmark di build/ricerca dell'indice (vedi sotto)
```

**Interfaccia web** (`cognivore serve`, poi apri l'URL indicato):
un'interfaccia di chat con streaming live dei token, una zona
drag-and-drop per l'ingestione di file `.txt`/`.md` e per file
audio/video, un selettore di lingua (9 lingue), tre temi
(scuro/chiaro/aurora) e una vista di traccia di ogni chiamata a
strumento fatta dall'agente per una data risposta (cliccando su un
passaggio si vede l'output completo e non troncato dello strumento).

**API REST/SSE**, una volta che `cognivore serve` è in esecuzione:

```bash
curl http://127.0.0.1:8420/api/health

curl -X POST http://127.0.0.1:8420/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is 12 * 7?"}'

curl -N "http://127.0.0.1:8420/api/chat/stream?message=Summarize+the+ingested+docs"

curl -X POST http://127.0.0.1:8420/api/ingest/file -F "file=@./notes.md"
```

**Come libreria**, piuttosto che tramite CLI o API (vedi `examples/`):

```python
from cognivore.bootstrap import build_agent
from cognivore.config import get_settings

agent = build_agent(get_settings())
result = agent.run("What's 15% of 5000?")
print(result.answer)
```

## Benchmark

Numeri da `python benchmarks/bench_index.py` su una macchina di classe
CI a 2 core (l'immagine del `Dockerfile` è portabile; il vostro laptop
reale ha quasi certamente più core, il che conta -- vedi sotto). Vettori
a 384 dimensioni, casuali e distribuiti uniformemente, il che è vicino a
un *caso peggiore* per la ricerca approssimata (nessuna struttura a
cluster reale, quindi i vicini "più prossimi" sono solo marginalmente
più vicini di vicini scelti a caso); con embedding reali il recall è
sensibilmente migliore allo stesso `ef`.

**Tempo di build** (3,000 vettori) -- qui il vantaggio dell'estensione
nativa è inequivocabile:

| Indice | Tempo di build | vs. fallback NumPy |
|---|---|---|
| `FlatIndexPy` (NumPy, `.add()` in un ciclo) | 0.615s | 1x |
| `FlatIndex` (C++) | 0.027s | **~23x più veloce** |

**Recall e velocità sono un parametro configurabile, non un numero
unico** (`NSWIndex`, n=3,000):

| `ef` | ms/query | Recall@10 |
|---|---|---|
| 50  | 0.24 | 0.64 |
| 150 (predefinito) | 0.46 | 0.93 |
| 400 | 0.78 | 1.00 |

**Latenza di ricerca in funzione della dimensione della collezione** --
una scansione lineare esatta (anche se accelerata con AVX2+OpenMP) è
*adeguata* finché la collezione non diventa così grande che la scansione
stessa diventa il collo di bottiglia; è in quel punto di incrocio che un
indice a grafo approssimato dovrebbe cominciare a vincere:

| n | `FlatIndex` ms/query | `NSWIndex` ms/query (ef=150) | Accelerazione |
|---|---|---|---|
| 1,000 | 0.036 | 0.240 | 0.2x (la forza bruta vince -- collezione troppo piccola) |
| 10,000 | 0.340 | 0.798 | 0.4x |
| 50,000 | 1.561 | 1.291 | 1.2x |

Leggete questa tabella per quello che dice realmente, non per quello che
farebbe una storia più bella: a queste dimensioni, su 2 core, una
scansione a forza bruta SIMD+parallela è *competitiva con, o più veloce
di,* l'indice approssimato scritto da zero. Questo è un comportamento
ANN reale e ben documentato -- un grafo NSW a singolo livello scritto a
mano ha un overhead per passo significativamente maggiore (operazioni
sullo heap, accesso casuale alla memoria attraverso il grafo, gestione
dell'insieme dei nodi visitati) di quanto ne goda una scansione lineare
cache-friendly, e l'HNSW multilivello (l'approccio di Faiss/hnswlib, non
ancora implementato qui -- vedi la roadmap) esiste specificamente per
ampliare questo scarto alla scala. La conclusione onesta: l'indice ANN
di questo progetto dimostra correttamente la *struttura dati e
l'algoritmo* (vedi il test di regressione del recall in
`tests/test_index.py`), e il punto di incrocio in cui inizia a ripagarsi
si sposta con il numero di core, con `ef`, e con quanto sono
clusterizzati i vostri embedding reali -- non è un'affermazione
universale del tipo "sempre più veloce", e questo README non
pretenderà che lo sia.

## Test

```bash
pip install -e ".[dev]"
pytest --cov                 # 71 test: sicurezza della calcolatrice, chunking,
                              # parità tra indice nativo e Python, recall NSW,
                              # ciclo dell'agente, store RAG, endpoint FastAPI
ruff check . && ruff format --check .
mypy -p cognivore
```

Ogni controllo elencato sopra è esattamente ciò che esegue la CI
(`.github/workflows/ci.yml`), su Ubuntu/macOS/Windows e Python
3.10-3.12; i test che dipendono solo dall'estensione nativa vengono
saltati automaticamente (invece di fallire) sulle configurazioni della
matrice in cui il toolchain C++ non è disponibile, replicando lo stesso
comportamento di fallback che ha il pacchetto stesso.

## Struttura del progetto

Vedi [docs/architecture.md](../architecture.md) (in inglese) per uno
schema e le motivazioni dietro le due decisioni di design più
importanti (un ciclo ReAct basato su testo invece del function calling
specifico del provider, e un indice scritto a mano invece di
Faiss/hnswlib).

```
native/            C++ vector index core + pybind11 bindings
src/cognivore/
  index/            native-vs-fallback index selection
  rag/              chunking, embeddings, hybrid document store
  agent/            ReAct loop, memory, prompt/parsing
  tools/            calculator, RAG search, audio, video
  llm/              llama.cpp backend + FakeLLMBackend
  media/            faster-whisper / OpenCV wrappers
  web/              FastAPI app + static chat UI
  cli.py            cognivore chat|ingest|serve|bench
benchmarks/         standalone scripts for the numbers above
examples/           minimal library-usage scripts
tests/              pytest suite (71 tests)
```

## Licenza

MIT – vedi [LICENSE](../../LICENSE).

## Contatti

Creato da **Sergeev Anton** ([GitHub](https://github.com/Anton-Sergeev-EA), [avsergeev1981@gmail.com](mailto:avsergeev1981@gmail.com)).
Contributi benvenuti – vedi [CONTRIBUTING.md](../../CONTRIBUTING.md).

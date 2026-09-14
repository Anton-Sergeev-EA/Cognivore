# Cognivore

**Ein lokal-first, torch-freies multimodales Agenten-Framework.** LLM-Reasoning
+ RAG, mit einem handgeschriebenen C++-Vektorindex und einer Web-Chat-UI --
alles läuft auf einem reinen CPU-Laptop, nichts muss Ihren Rechner verlassen.

[![CI](https://github.com/Anton-Sergeev-EA/cognivore/actions/workflows/ci.yml/badge.svg)](https://github.com/Anton-Sergeev-EA/cognivore/actions/workflows/ci.yml)
[![CodeQL](https://github.com/Anton-Sergeev-EA/cognivore/actions/workflows/codeql.yml/badge.svg)](https://github.com/Anton-Sergeev-EA/cognivore/actions/workflows/codeql.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](../../LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](../../pyproject.toml)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-black)](https://github.com/astral-sh/ruff)

🌐 **Lies dies in einer anderen Sprache:** [English](../../README.md) | [Русский](README.ru.md) | [Deutsch](README.de.md) | [Français](README.fr.md) | [Italiano](README.it.md) | [Español](README.es.md) | [中文](README.zh.md) | [日本語](README.ja.md) | [हिन्दी](README.hi.md)

Cognivore ist ein Agenten-Framework im ReAct-Stil mit retrieval-augmented
generation und Unterstützung für multimodale Werkzeuge (Audio-Transkription,
Video-Szenenanalyse), das gezielt so gebaut wurde, dass es vollständig auf
einer reinen CPU-Maschine läuft, ohne irgendeine PyTorch-Abhängigkeit im
gesamten Stack. Sein Retrieval-Index ist ein von Grund auf handgeschriebener
C++-Kern (AVX2 SIMD, OpenMP-paralleler exakter Suchlauf und ein selbst
entwickelter approximativer NSW-Graph), der über pybind11 an Python
angebunden ist, mit einem reinen NumPy-Fallback, sodass `pip install` niemals
allein wegen eines fehlenden C++-Compilers scheitert.

```
you> What does the ingested spec say about the retry policy, and what's 15% of that timeout in ms?

  [tool] search_knowledge_base({"query": "retry policy timeout"})
  [tool] calculator({"expression": "5000 * 0.15"})

cognivore> The spec (spec.md) sets a 5000ms timeout with exponential
backoff. 15% of that is 750ms.
```

Derselbe Ablauf -- in der Web-UI: eine Live-Trace des Tool-Aufrufs und der gefundenen Passage, in jeder der 9 Sprachen und jedem der 3 Themes:

| Aurora-Theme, Englisch -- RAG-Trace | Dunkles Theme, Russisch -- Taschenrechner |
|---|---|
| ![Cognivore Web-UI: Aurora-Theme, Englisch, ein search_knowledge_base-Tool-Aufruf und die gefundene Passage](../screenshots/web-ui-en.png) | ![Cognivore Web-UI: dunkles Theme, Russisch, ein calculator-Tool-Aufruf](../screenshots/web-ui-ru.png) |

## Warum es dieses Projekt gibt

Die meisten "AI-Agenten"-Portfolio-Projekte sind ein dünner Wrapper um einen
API-Aufruf. Dieses Projekt zeigt das Gegenteil: ein Framework, in dem die
interessanten Teile (eine echte ANN-Datenstruktur, ein provider-unabhängiges
Tool-Calling-Protokoll, eine hybride Retrieval-Pipeline, ein geordneter
Rückfall, wenn optionale Abhängigkeiten fehlen) tatsächlich implementiert
und nicht nur importiert sind.

## Funktionen

- **ReAct-Agentenschleife** (Thought → Action → Observation), die mit
  *jedem* instruction-getunten lokalen Modell funktioniert, nicht nur mit
  solchen, die auf ein bestimmtes Function-Calling-Wire-Format feinabgestimmt
  wurden -- siehe
  [docs/architecture.md](../architecture.md#why-a-react-loop-instead-of-native-function-calling).
- **Nativer C++-Vektorindex** (`native/vector_index.cpp`): ein exakter
  `FlatIndex` (AVX2/FMA-Skalarprodukt, OpenMP-paralleler Scan) und ein
  approximativer `NSWIndex` (ein von Grund auf selbst entwickelter
  einschichtiger Navigable-Small-World-Graph -- Einfügen, Nachbar-Pruning,
  gierige Beam-Suche), beide über eine handgeschriebene pybind11-Erweiterung
  mit einem `.pyi`-Type-Stub an Python angebunden. Fällt automatisch auf
  einen reinen NumPy-Index zurück, falls zur Installationszeit kein
  C++-Compiler vorhanden ist.
- **Hybrides RAG**: Chunking über einen rekursiven Splitter, semantische
  `fastembed`-Embeddings (ONNX, ohne PyTorch) mit einem Hashing-Trick-Fallback
  ohne jeglichen Download, hybrides Retrieval aus Vektor- und BM25-Suche.
- **Multimodale Werkzeuge**: ein sicherer (AST-basierter, ohne `eval`)
  Rechner, Wissensdatenbank-Suche, Audio-Transkription mit grober
  Sprecherzuordnung (`faster-whisper`) und Video-Szenenerkennung (OpenCV)
  + Texterkennung im Bild per OCR (Tesseract) -- alles rein CPU-basiert,
  ohne PyTorch.
- **Austauschbares LLM-Backend**: lokale GGUF-Inferenz über
  `llama-cpp-python`, oder ein deterministisches, abhängigkeitsfreies
  `FakeLLMBackend`, das exakt denselben Tool-Calling-Codepfad durchläuft,
  ganz ohne Download -- genau darauf laufen die Testsuite und CI.
- **Web-UI**: FastAPI + SSE-Streaming + eine Chat-Oberfläche aus reinem
  JS/HTML/CSS (kein Build-Schritt, kein Framework) mit Drag-and-Drop für
  Datei-/Audio-/Video-Uploads, Live-Indikatoren für "Denkt nach"/Verbindung,
  Dark/Light/Aurora-Themes und Lokalisierung ins Englische, Russische,
  Deutsche, Französische, Italienische, Spanische, vereinfachte Chinesische,
  Japanische und Hindi.
- Vollständige Testsuite, ruff-Linting+Formatierung, mypy (weitgehend
  strikt, inklusive eines Stubs für die native Erweiterung) und eine
  Multi-OS-/Multi-Python-CI-Matrix.

## Schnellstart

```bash
git clone https://github.com/Anton-Sergeev-EA/cognivore.git
cd cognivore
python3 -m venv .venv && source .venv/bin/activate
pip install -e .                 # baut die native Erweiterung, falls ein C++17-Compiler vorhanden ist

cognivore chat                   # interaktiver Chat, standardmäßig im Offline-Demomodus
cognivore ingest ./docs          # indexiert einen Ordner mit Markdown-/Textdateien
cognivore serve                  # Web-Chat-UI unter http://127.0.0.1:8420
```

Standardmäßig wird keine LLM heruntergeladen und es findet überhaupt kein
Netzwerkzugriff statt: Der Agent läuft gegen `FakeLLMBackend`, ein kleines
deterministisches Backend, das trotzdem die reale Tool-Calling-Schleife
durchläuft (probieren Sie `What is 12 * 7?` oder `search the knowledge base
for ...`, nachdem Sie etwas mit `ingest` verarbeitet haben). Um eine echte
lokale LLM zu verwenden:

```bash
pip install -e ".[llm]"
# laden Sie z. B. https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF herunter
cp .env.example .env
echo 'COGNIVORE_LLM_MODEL_PATH=/path/to/model.gguf' >> .env
cognivore chat
```

Oder, einfacher und ganz ohne C++-Toolchain, verwenden Sie statt dessen
[Ollama](https://ollama.com) -- `COGNIVORE_LLM_PROVIDER=auto` (der
Standardwert) versucht zuerst einen lokal laufenden Ollama-Server, bevor es
auf einen GGUF-Pfad oder `FakeLLMBackend` zurückfällt:

```bash
ollama pull qwen2.5:3b   # jedes instruction-getunte Modell funktioniert
cognivore chat           # nimmt den laufenden Ollama-Server automatisch auf
```

Audio-/Video-Werkzeuge benötigen ihre eigenen Extras:
`pip install -e ".[audio,video]"` (oder `.[all]` für alles zusammen,
inklusive GGUF). Alle Einstellungen finden Sie in `.env.example`.

Die Texterkennung im Video-Werkzeug (`analyze_video`) benötigt zusätzlich
die [Tesseract](https://github.com/tesseract-ocr/tesseract)-OCR-*Binary*
selbst -- das über das `video`-Extra installierte `pytesseract`-Paket ist
nur ein dünner Wrapper darum und liefert ohne die Binary stillschweigend
keinen Text (Szenenerkennung und Zeitstempel funktionieren trotzdem, da
dieser Teil rein auf OpenCV beruht). Standardmäßig erkennt sie Englisch
*und* Russisch (`eng+rus`, siehe `COGNIVORE_OCR_LANGUAGES` in
`.env.example`) -- auf Debian/Ubuntu zieht das einfache Paket
`tesseract-ocr` automatisch `eng` nach, aber nicht `rus`, deshalb beide
explizit installieren:

```bash
sudo apt install tesseract-ocr tesseract-ocr-rus   # Debian/Ubuntu
brew install tesseract                              # macOS -- liefert alle Sprachen zusammen
# Windows: https://github.com/UB-Mannheim/tesseract/wiki (in der Sprachliste des Installers Russisch ankreuzen)
```

Das Docker-Image bringt beide bereits mit -- dort ist nichts zu installieren.

Fordert man eine Sprache an, deren Trainingsdaten-Paket nicht installiert
ist, gibt es keinen Fehler -- die Schrift wird stillschweigend als
ähnlich aussehende lateinische Buchstaben fehlerkannt (aus dem
kyrillischen "Контейнеры" wird "KoHTewHepbi"), was eher wie ein
schlechter Scan als wie ein fehlendes Sprachpaket wirkt. Verwendet Ihr
Bildschirmtext eine andere Sprache, installieren Sie deren Paket
`tesseract-ocr-<lang>` und fügen Sie sie zu `COGNIVORE_OCR_LANGUAGES`
hinzu (z. B. `eng+rus+deu`).

### Docker

Das Image wird in mehreren Stufen gebaut (kompiliert die native
C++-Erweiterung und verwirft den Compiler anschließend für ein schlankes
Runtime-Image), läuft als Non-Root-User und liefert einen `HEALTHCHECK` mit --
einmal gebaut mit dem Docker-Layer-Caching von GitHub Actions und bei jedem
Push end-to-end verifiziert (der Server antwortet tatsächlich auf
`/api/health`, der Container meldet `healthy`, läuft als Non-Root) -- es ist
also nicht nur "baut durch", sondern "geprüft".

**Der einfachste Weg -- kein Python, keine Ollama-Installation, funktioniert
gleich auf Windows, macOS und Linux:**

```bash
docker compose up -d --build
docker compose exec ollama ollama pull qwen2.5:3b   # einmalig, ~2 GB
```

Öffnen Sie dann <http://127.0.0.1:8420>. `docker-compose.yml` startet Ollama
ebenfalls *in seinem eigenen* Container, sodass auf dem Host außer Docker
selbst nichts installiert werden muss; Cognivore erreicht ihn über das
Compose-Netzwerk anhand des Servicenamens (`http://ollama:11434`), was die
üblichen Unterschiede im Host-Networking zwischen Windows/macOS/Linux
vollständig umschifft. Ohne dieses einmalige `ollama pull` startet Cognivore
trotzdem einwandfrei -- es fällt lediglich auf den offline-Demomodus von
`FakeLLMBackend` zurück, bis ein Modell verfügbar ist.

Der Compose-Stack setzt außerdem `COGNIVORE_SEED_DEMO_KB=true`, sodass eine
frische Wissensdatenbank automatisch mit zwei mitgelieferten
Demo-Firmenhandbüchern (Englisch + Russisch -- Preise, SLA, Sicherheit,
Rückerstattungsrichtlinie, Support-FAQ) befüllt wird, statt eine leere
Dropzone zu zeigen. Es wird ausschließlich eine *leere* Wissensdatenbank
befüllt: Sobald Sie eigene Dokumente eingelesen haben, ist dies dauerhaft ein
No-op. Setzen Sie es in `docker-compose.yml` auf `false` (oder `docker run -e
COGNIVORE_SEED_DEMO_KB=false`), um leer zu starten, oder führen Sie jederzeit
`cognivore seed-demo` aus, um dieselben Demo-Dokumente nachträglich zu einem
bestehenden Store hinzuzufügen.

**Haben Sie bereits Ollama auf dem Host laufen, oder möchten Sie nur einen
einzigen Container?**

```bash
docker build -t cognivore .
docker run -d -p 8420:8420 -v cognivore-data:/data \
  -e COGNIVORE_OLLAMA_HOST=http://host.docker.internal:11434 \
  --add-host=host.docker.internal:host-gateway \
  cognivore
```

`host.docker.internal` wird auf Docker Desktop (Windows/macOS) automatisch
bereitgestellt; das explizite `--add-host` oben ist das, was denselben
Befehl auch auf einem gewöhnlichen Linux-System funktionsfähig macht, wo der
Name sonst nicht aufgelöst würde.

Das Image liefert die Audio-/Video-Werkzeuge (`faster-whisper`, OpenCV, und
Tesseract für die Texterkennung im Bild) mit, aber *nicht* `llama-cpp-python` -- es spricht stattdessen bewusst über
gewöhnliches HTTP mit Ollama, statt eine GGUF-Datei im selben Prozess zu
laden, da llama-cpp-python nicht für jede Plattform ein vorgefertigtes Wheel
hat und einen Compiler benötigt, den die Runtime-Stufe nicht mitbringt.
Möchten Sie trotzdem In-Process-GGUF-Inferenz innerhalb des Containers?
Fügen Sie `build-essential` zur finalen Stufe hinzu und stellen Sie deren
`pip install` wieder auf das Extra `[all]` um.

`faster-whisper` lädt sein Spracherkennungsmodell erst beim ersten tatsächlichen
Einsatz der Audiotranskription von Hugging Face herunter (nicht beim Build) –
genauso wie `ollama pull` für das LLM, nur dass dies hier automatisch beim
ersten Gebrauch geschieht, statt einen expliziten Befehl zu erfordern. Wie die
Modelle von Ollama wird es im persistenten Volume `cognivore-data`
zwischengespeichert (`HF_HOME=/data/hf-cache`), sodass es nur einmal
heruntergeladen wird und nicht bei jedem `docker compose up --build`.

**Stoppen und neu starten** (z. B. nach einem Neustart):

```bash
docker compose down   # stoppt beide Container; Daten bleiben erhalten (siehe unten)
docker compose up -d  # startet erneut -- kein --build nötig, außer das Image selbst hat sich geändert
```

Beide Dienste sind auf `restart: unless-stopped` gesetzt. Wenn Sie sie also vor
dem Herunterfahren nicht manuell gestoppt haben, startet Docker sie von selbst
neu, sobald der Docker-Daemon wieder hochkommt (die Standardeinstellung bei den
meisten Installationen) -- in diesem Fall ist überhaupt kein Befehl nötig. Die
Wissensdatenbank und die zwischengespeicherten Whisper-/Ollama-Modelle liegen
in den benannten Volumes `cognivore-data` und `ollama-data`, die
`docker compose down` niemals berührt; nur ein explizites
`docker compose down -v` entfernt sie.

**Vorgefertigtes Image (überhaupt kein Build-Schritt):** getaggte Releases
werden multi-arch (amd64 + arm64 -- Apple Silicon und Raspberry Pi
eingeschlossen) über
[`.github/workflows/docker-publish.yml`](../../.github/workflows/docker-publish.yml)
nach GHCR veröffentlicht:

```bash
docker pull ghcr.io/anton-sergeev-ea/cognivore:latest
```

## Verwendung

**CLI** (`cognivore --help` für die vollständige Liste):

```bash
cognivore chat                              # interaktive REPL
cognivore ingest ./docs                     # indexiert rekursiv .txt/.md/.markdown/.rst
cognivore seed-demo                         # fügt die mitgelieferten EN+RU-Demo-Firmenhandbücher hinzu
cognivore serve --host 0.0.0.0 --port 8420  # Web-UI + REST/SSE-API
cognivore bench                             # Benchmarks für Indexaufbau/-suche (siehe unten)
```

**Web-UI** (`cognivore serve`, dann die ausgegebene URL öffnen): eine
Chat-Oberfläche mit Live-Token-Streaming, einer Drag-and-Drop-Zone für
`.txt`/`.md`-Ingestion sowie für Audio-/Video-Dateien, einem Sprachumschalter
(9 Sprachen), drei Themes (Dark/Light/Aurora) und einer Trace-Ansicht jedes
Tool-Aufrufs, den der Agent für eine bestimmte Antwort gemacht hat (Klick
auf einen Schritt zeigt die vollständige, nicht abgeschnittene
Tool-Ausgabe).

**REST/SSE-API**, sobald `cognivore serve` läuft:

```bash
curl http://127.0.0.1:8420/api/health

curl -X POST http://127.0.0.1:8420/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is 12 * 7?"}'

curl -N "http://127.0.0.1:8420/api/chat/stream?message=Summarize+the+ingested+docs"

curl -X POST http://127.0.0.1:8420/api/ingest/file -F "file=@./notes.md"
```

**Als Bibliothek**, statt über die CLI oder die API (siehe `examples/`):

```python
from cognivore.bootstrap import build_agent
from cognivore.config import get_settings

agent = build_agent(get_settings())
result = agent.run("What's 15% of 5000?")
print(result.answer)
```

## Benchmarks

Zahlen aus `python benchmarks/bench_index.py` auf einer 2-Kern-Maschine der
CI-Klasse (das Image von `Dockerfile` ist portabel; Ihr tatsächlicher Laptop
hat mit ziemlicher Sicherheit mehr Kerne, was eine Rolle spielt -- siehe
unten). Zufällige, gleichmäßig verteilte 384-dimensionale Vektoren, was
nahe an einem *Worst Case* für die approximative Suche liegt (keine
tatsächliche Clusterstruktur, sodass die "nächsten" Nachbarn nur unwesentlich
näher sind als zufällige); echte Embeddings erreichen bei demselben `ef`
einen deutlich besseren Recall.

**Build-Zeit** (3.000 Vektoren) -- hier ist der Vorteil der nativen
Erweiterung eindeutig:

| Index | Build-Zeit | vs. NumPy-Fallback |
|---|---|---|
| `FlatIndexPy` (NumPy, `.add()` in einer Schleife) | 0.615s | 1x |
| `FlatIndex` (C++) | 0.027s | **~23x schneller** |

**Recall vs. Geschwindigkeit ist ein einstellbarer Regler, keine einzelne
Zahl** (`NSWIndex`, n=3.000):

| `ef` | ms/Anfrage | Recall@10 |
|---|---|---|
| 50  | 0.24 | 0.64 |
| 150 (Standard) | 0.46 | 0.93 |
| 400 | 0.78 | 1.00 |

**Suchlatenz vs. Kollektionsgröße** -- ein exakter linearer Scan (selbst mit
AVX2+OpenMP-Beschleunigung) ist *völlig in Ordnung*, solange die Kollektion
nicht so groß wird, dass das Scannen selbst zum Flaschenhals wird; genau an
diesem Übergangspunkt soll ein approximativer Graph-Index anfangen zu
gewinnen:

| n | `FlatIndex` ms/Anfrage | `NSWIndex` ms/Anfrage (ef=150) | Speedup |
|---|---|---|---|
| 1.000 | 0.036 | 0.240 | 0.2x (Brute-Force gewinnt -- Kollektion zu klein) |
| 10.000 | 0.340 | 0.798 | 0.4x |
| 50.000 | 1.561 | 1.291 | 1.2x |

Lesen Sie diese Tabelle so, wie sie tatsächlich ist, nicht so, wie es eine
schönere Geschichte ergäbe: bei diesen Größen, auf 2 Kernen, ist ein
SIMD+paralleler Brute-Force-Scan *konkurrenzfähig mit oder schneller als*
der von Grund auf selbst entwickelte approximative Index. Das ist reales,
gut dokumentiertes ANN-Verhalten -- ein handgeschriebener einschichtiger
NSW-Graph hat pro Schritt merklich mehr Overhead (Heap-Operationen,
wahlfreier Speicherzugriff über den Graphen, Buchführung über besuchte
Knoten) als ein Cache-freundlicher linearer Scan, und mehrschichtiges HNSW
(der Ansatz von Faiss/hnswlib, hier noch nicht implementiert -- siehe die
Roadmap) existiert gerade deshalb, um diese Lücke im großen Maßstab zu
vergrößern. Das ehrliche Fazit: Der ANN-Index dieses Projekts demonstriert
die *Datenstruktur und den Algorithmus* korrekt (siehe den
Recall-Regressionstest in `tests/test_index.py`), und der Punkt, an dem er
sich auszahlt, verschiebt sich mit der Kernanzahl, `ef` und wie stark Ihre
tatsächlichen Embeddings geclustert sind -- das ist keine universelle
Behauptung "immer schneller", und dieses README wird nicht so tun, als wäre
es eine.

## Tests

```bash
pip install -e ".[dev]"
pytest --cov                 # 71 Tests: Sicherheit des Rechners, Chunking,
                              # Parität von nativem und Python-Index, NSW-Recall,
                              # Agentenschleife, RAG-Store, FastAPI-Endpunkte
ruff check . && ruff format --check .
mypy -p cognivore
```

Alles oben Genannte ist genau das, was CI ausführt
(`.github/workflows/ci.yml`), über Ubuntu/macOS/Windows und Python
3.10-3.12 hinweg; Tests, die ausschließlich die native Erweiterung
betreffen, überspringen sich selbst (statt zu scheitern) auf einem
Matrix-Zweig, auf dem die C++-Toolchain nicht verfügbar ist -- genau wie
sich das Paket selbst verhält.

## Projektstruktur

Siehe [docs/architecture.md](../architecture.md) (auf Englisch) für ein
Diagramm und die Begründung hinter den beiden größten Architekturentscheidungen
(eine textbasierte ReAct-Schleife statt providerspezifischem Function
Calling, und ein handgeschriebener Index statt Faiss/hnswlib).

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

## Lizenz

MIT -- siehe [LICENSE](../../LICENSE).

## Kontakt

Entwickelt von **Sergeev Anton** ([GitHub](https://github.com/Anton-Sergeev-EA), [avsergeev1981@gmail.com](mailto:avsergeev1981@gmail.com)).
Beiträge sind willkommen -- siehe [CONTRIBUTING.md](../../CONTRIBUTING.md).

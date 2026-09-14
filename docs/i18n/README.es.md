# Cognivore

**Un framework de agentes multimodal local-first y sin PyTorch.** Razonamiento
LLM + RAG, con un índice vectorial escrito a mano en C++ y una interfaz de
chat web -- todo funciona en un portátil solo con CPU, nada tiene que salir
de tu máquina.

[![CI](https://github.com/Anton-Sergeev-EA/cognivore/actions/workflows/ci.yml/badge.svg)](https://github.com/Anton-Sergeev-EA/cognivore/actions/workflows/ci.yml)
[![CodeQL](https://github.com/Anton-Sergeev-EA/cognivore/actions/workflows/codeql.yml/badge.svg)](https://github.com/Anton-Sergeev-EA/cognivore/actions/workflows/codeql.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](../../LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](../../pyproject.toml)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-black)](https://github.com/astral-sh/ruff)

🌐 **Lee esto en otro idioma:** [English](../../README.md) | [Русский](README.ru.md) | [Deutsch](README.de.md) | [Français](README.fr.md) | [Italiano](README.it.md) | [Español](README.es.md) | [中文](README.zh.md) | [日本語](README.ja.md) | [हिन्दी](README.hi.md)

Cognivore es un framework de agentes al estilo ReAct con retrieval-augmented
generation y soporte para herramientas multimodales (transcripción de audio,
análisis de escenas en vídeo), construido específicamente para funcionar
por completo en una máquina solo con CPU, sin ninguna dependencia de
PyTorch en toda la pila. Su índice de recuperación es un núcleo en C++
escrito desde cero (SIMD con AVX2, búsqueda exacta paralelizada con OpenMP y
un grafo NSW aproximado también desde cero), expuesto a Python mediante
pybind11, con un fallback en NumPy puro para que `pip install` nunca falle
por falta de un compilador de C++.

```
you> What does the ingested spec say about the retry policy, and what's 15% of that timeout in ms?

  [tool] search_knowledge_base({"query": "retry policy timeout"})
  [tool] calculator({"expression": "5000 * 0.15"})

cognivore> The spec (spec.md) sets a 5000ms timeout with exponential
backoff. 15% of that is 750ms.
```

El mismo ciclo, en la interfaz web -- una traza en vivo de la llamada a la herramienta y el fragmento recuperado, en cualquiera de los 9 idiomas y los 3 temas:

| Tema aurora, inglés -- traza de RAG | Tema oscuro, ruso -- calculadora |
|---|---|
| ![Interfaz web de Cognivore: tema aurora, inglés, una llamada a la herramienta search_knowledge_base y el fragmento recuperado](../screenshots/web-ui-en.png) | ![Interfaz web de Cognivore: tema oscuro, ruso, una llamada a la herramienta calculator](../screenshots/web-ui-ru.png) |

## Por qué existe este proyecto

La mayoría de los proyectos de portafolio de "agentes de IA" son una fina
envoltura sobre una llamada a una API. Este proyecto muestra lo contrario:
un framework donde las partes interesantes (una estructura de datos ANN
real, un protocolo de invocación de herramientas independiente del
proveedor, un pipeline de recuperación híbrido, una degradación elegante
cuando faltan dependencias opcionales) están implementadas, no importadas.

## Características

- **Bucle de agente ReAct** (Thought → Action → Observation) que funciona
  con *cualquier* modelo local instruction-tuned, no solo con modelos
  afinados para un formato de function calling específico -- ver
  [docs/architecture.md](../architecture.md#why-a-react-loop-instead-of-native-function-calling).
- **Índice vectorial nativo en C++** (`native/vector_index.cpp`): un
  `FlatIndex` exacto (producto punto con AVX2/FMA, escaneo paralelizado con
  OpenMP) y un `NSWIndex` aproximado (un grafo Navigable Small World de una
  sola capa escrito desde cero -- inserción, poda de vecinos, búsqueda greedy
  por haz), ambos vinculados a Python mediante una extensión pybind11
  escrita a mano con un stub de tipos `.pyi`. Recurre automáticamente a un
  índice en NumPy puro si no hay compilador de C++ disponible en el momento
  de la instalación.
- **RAG híbrido**: fragmentación (chunking) con un splitter recursivo,
  embeddings semánticos con `fastembed` (ONNX, sin PyTorch) con un fallback
  de hashing-trick sin descargas, recuperación híbrida vector + BM25.
- **Herramientas multimodales**: calculadora segura (basada en AST, sin
  `eval`), búsqueda en la base de conocimiento, transcripción de audio con
  segmentación aproximada por hablante (`faster-whisper`), y detección de
  escenas en vídeo (OpenCV) + OCR de texto en pantalla (Tesseract) -- todo
  solo con CPU, sin PyTorch.
- **Backend de LLM enchufable**: inferencia local GGUF mediante
  `llama-cpp-python`, o un `FakeLLMBackend` determinista y sin
  dependencias que ejerce exactamente la misma ruta de código de invocación
  de herramientas sin ninguna descarga -- es contra lo que corren la
  suite de tests y la CI.
- **Interfaz web**: FastAPI + streaming por SSE + una interfaz de chat en
  JS/HTML/CSS puro (sin paso de build, sin framework) con arrastrar y
  soltar archivos/audio/vídeo, indicadores en vivo de "pensando"/conexión,
  temas oscuro/claro/aurora, y localización al inglés, ruso, alemán,
  francés, italiano, español, chino simplificado, japonés e hindi.
- Suite de tests completa, lint+formato con ruff, mypy (bastante estricto,
  incluyendo un stub para la extensión nativa) y una matriz de CI
  multi-SO/multi-Python.

## Inicio rápido

```bash
git clone https://github.com/Anton-Sergeev-EA/cognivore.git
cd cognivore
python3 -m venv .venv && source .venv/bin/activate
pip install -e .                 # builds the native extension if a C++17 compiler is present

cognivore chat                   # interactive chat, offline demo mode by default
cognivore ingest ./docs          # index a folder of markdown/text files
cognivore serve                  # web chat UI at http://127.0.0.1:8420
```

Por defecto no hay ninguna descarga de LLM ni acceso a red en absoluto: el
agente funciona contra `FakeLLMBackend`, un backend pequeño y determinista
que aun así ejerce el bucle real de invocación de herramientas (prueba
`What is 12 * 7?` o `search the knowledge base for ...` después de hacer
`ingest` de algo). Para usar una LLM local real:

```bash
pip install -e ".[llm]"
# download e.g. https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF
cp .env.example .env
echo 'COGNIVORE_LLM_MODEL_PATH=/path/to/model.gguf' >> .env
cognivore chat
```

O, más sencillo aún y sin necesidad de ninguna cadena de compilación de
C++, apúntalo en su lugar a [Ollama](https://ollama.com) --
`COGNIVORE_LLM_PROVIDER=auto` (el valor por defecto) primero intenta un
servidor Ollama en ejecución local antes de recurrir a una ruta GGUF o a
`FakeLLMBackend`:

```bash
ollama pull qwen2.5:3b   # any instruction-tuned model works
cognivore chat           # picks up the running Ollama server automatically
```

Las herramientas de audio/vídeo necesitan sus propios extras:
`pip install -e ".[audio,video]"` (o `.[all]` para todo, GGUF incluido).
Consulta `.env.example` para ver todas las opciones de configuración.

La extracción de texto en pantalla de la herramienta de vídeo
(`analyze_video`) necesita además el *binario* de OCR de
[Tesseract](https://github.com/tesseract-ocr/tesseract) -- el paquete
`pytesseract` que trae el extra `video` no es más que un envoltorio fino
alrededor de él, y sin él el OCR no devuelve texto silenciosamente (la
detección de escenas y las marcas de tiempo funcionan igual, ya que esa
parte es puro OpenCV):

```bash
sudo apt install tesseract-ocr        # Debian/Ubuntu
brew install tesseract                # macOS
# Windows: https://github.com/UB-Mannheim/tesseract/wiki
```

La imagen de Docker ya lo incluye -- no hay nada que instalar ahí.

### Docker

La imagen se construye en varias etapas (compila la extensión nativa en
C++, luego descarta el compilador para obtener una imagen de runtime
ligera), se ejecuta como usuario sin privilegios de root, e incluye un
`HEALTHCHECK` -- se construye una vez aprovechando el cacheo de capas de
Docker de GitHub Actions y se verifica de extremo a extremo (el servidor
responde realmente en `/api/health`, el contenedor reporta `healthy`, se
ejecuta sin root) en cada push, así que no es solo que "compila", es que
está verificada.

**El camino más sencillo -- sin Python, sin instalar Ollama, funciona igual
en Windows, macOS y Linux:**

```bash
docker compose up -d --build
docker compose exec ollama ollama pull qwen2.5:3b   # one-time, ~2GB
```

Luego abre <http://127.0.0.1:8420>. `docker-compose.yml` ejecuta Ollama
*también* dentro de su propio contenedor, así que no hay nada que instalar
en el host más allá del propio Docker; Cognivore lo alcanza a través de la
red de compose por el nombre del servicio (`http://ollama:11434`), lo que
evita por completo las diferencias habituales de redes de host entre
Windows/macOS/Linux. Sin ese `ollama pull` inicial, Cognivore de todos
modos arranca y funciona correctamente -- simplemente recae en el modo de
demostración offline `FakeLLMBackend` hasta que haya un modelo disponible.

El stack de compose también define `COGNIVORE_SEED_DEMO_KB=true`, por lo que
una base de conocimiento nueva se llena automáticamente con dos manuales de
empresa de demostración incluidos (inglés + ruso -- precios, SLA, seguridad,
política de reembolsos, preguntas frecuentes de soporte) en lugar de abrir con
una zona de carga vacía. Solo llena una base de conocimiento *vacía*: en
cuanto hayas ingerido tus propios documentos, esto se convierte en un no-op
permanente. Ponlo en `false` en `docker-compose.yml` (o `docker run -e
COGNIVORE_SEED_DEMO_KB=false`) para empezar vacío, o ejecuta `cognivore
seed-demo` en cualquier momento para añadir los mismos documentos de
demostración a un almacén ya existente.

**¿Ya tienes Ollama corriendo en el host, o quieres un único contenedor?**

```bash
docker build -t cognivore .
docker run -d -p 8420:8420 -v cognivore-data:/data \
  -e COGNIVORE_OLLAMA_HOST=http://host.docker.internal:11434 \
  --add-host=host.docker.internal:host-gateway \
  cognivore
```

`host.docker.internal` se proporciona automáticamente en Docker Desktop
(Windows/macOS); el `--add-host` explícito de arriba es lo que hace que
ese mismo comando también funcione en Linux tal cual, donde de otro modo
ese nombre no se resolvería.

La imagen incluye las herramientas de audio/vídeo (`faster-whisper`,
OpenCV, y Tesseract para el OCR de texto en pantalla) pero *no*
`llama-cpp-python` -- se comunica con Ollama por HTTP
normal en lugar de cargar un archivo GGUF en el propio proceso,
deliberadamente, ya que `llama-cpp-python` no tiene una wheel prebuilt para
todas las plataformas y necesita un compilador que la etapa de runtime no
lleva. ¿Aun así quieres inferencia GGUF en proceso dentro del contenedor?
Añade `build-essential` a la etapa final y cambia su `pip install` de
vuelta al extra `[all]`.

`faster-whisper` descarga su modelo de reconocimiento de voz desde Hugging
Face la primera vez que la transcripción de audio se usa realmente (no en
tiempo de build), igual que `ollama pull` para el LLM -- la diferencia es
que esto ocurre automáticamente en el primer uso, sin necesidad de un
comando explícito. Igual que los modelos de Ollama, se cachea en el volumen
persistido `cognivore-data` (`HF_HOME=/data/hf-cache`), así que solo se
descarga una vez, no en cada `docker compose up --build`.

**Imagen preconstruida (sin ningún paso de build):** las releases
etiquetadas se publican multi-arquitectura (amd64 + arm64 -- incluyendo
Apple Silicon y Raspberry Pi) en GHCR mediante
[`.github/workflows/docker-publish.yml`](../../.github/workflows/docker-publish.yml):

```bash
docker pull ghcr.io/anton-sergeev-ea/cognivore:latest
```

## Uso

**CLI** (`cognivore --help` para la lista completa):

```bash
cognivore chat                              # interactive REPL
cognivore ingest ./docs                     # recursively indexes .txt/.md/.markdown/.rst
cognivore seed-demo                         # añade los manuales de empresa demo EN+RU incluidos
cognivore serve --host 0.0.0.0 --port 8420  # web UI + REST/SSE API
cognivore bench                             # index build/search benchmarks (see below)
```

**Interfaz web** (`cognivore serve`, luego abre la URL que se imprime): una
interfaz de chat con streaming de tokens en vivo, una zona de arrastrar y
soltar para ingerir archivos `.txt`/`.md` y para archivos de audio/vídeo,
un selector de idioma (9 idiomas), tres temas (oscuro/claro/aurora), y una
vista de traza de cada llamada a herramienta que hizo el agente para una
respuesta dada (haz clic en un paso para ver la salida completa y sin
truncar de la herramienta).

**API REST/SSE**, una vez que `cognivore serve` está en ejecución:

```bash
curl http://127.0.0.1:8420/api/health

curl -X POST http://127.0.0.1:8420/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is 12 * 7?"}'

curl -N "http://127.0.0.1:8420/api/chat/stream?message=Summarize+the+ingested+docs"

curl -X POST http://127.0.0.1:8420/api/ingest/file -F "file=@./notes.md"
```

**Como librería**, en lugar de a través de la CLI o de la API (ver
`examples/`):

```python
from cognivore.bootstrap import build_agent
from cognivore.config import get_settings

agent = build_agent(get_settings())
result = agent.run("What's 15% of 5000?")
print(result.answer)
```

## Benchmarks

Cifras de `python benchmarks/bench_index.py` en una máquina de 2 núcleos
de clase CI (la imagen del `Dockerfile` es portable; tu portátil real casi
con toda seguridad tiene más núcleos, y eso importa -- ver más abajo).
Vectores aleatorios, uniformemente distribuidos, de 384 dimensiones, lo
cual está cerca de ser el *peor caso* para la búsqueda aproximada (no hay
una estructura de clústeres real, así que los vecinos "más cercanos" solo
son marginalmente más cercanos que unos aleatorios); con embeddings reales
el recall es notablemente mejor con el mismo `ef`.

**Tiempo de construcción** (3.000 vectores) -- aquí la ventaja de la
extensión nativa es inequívoca:

| Índice | Tiempo de construcción | vs. fallback en NumPy |
|---|---|---|
| `FlatIndexPy` (NumPy, `.add()` en un bucle) | 0.615s | 1x |
| `FlatIndex` (C++) | 0.027s | **~23x más rápido** |

**El recall frente a la velocidad es un parámetro ajustable, no un único
número** (`NSWIndex`, n=3.000):

| `ef` | ms/consulta | Recall@10 |
|---|---|---|
| 50  | 0.24 | 0.64 |
| 150 (por defecto) | 0.46 | 0.93 |
| 400 | 0.78 | 1.00 |

**Latencia de búsqueda frente al tamaño de la colección** -- un escaneo
lineal exacto (incluso acelerado con AVX2+OpenMP) va *bien* hasta que la
colección es lo bastante grande como para que escanearla se convierta en
el cuello de botella; ese punto de cruce es justo donde un índice de grafo
aproximado se supone que empieza a ganar:

| n | `FlatIndex` ms/consulta | `NSWIndex` ms/consulta (ef=150) | Aceleración |
|---|---|---|---|
| 1.000 | 0.036 | 0.240 | 0.2x (gana la fuerza bruta -- colección demasiado pequeña) |
| 10.000 | 0.340 | 0.798 | 0.4x |
| 50.000 | 1.561 | 1.291 | 1.2x |

Lee esa tabla por lo que realmente dice, no por lo que haría una historia
más bonita: a estos tamaños, en 2 núcleos, un escaneo de fuerza bruta con
SIMD y paralelo es *competitivo o más rápido* que el índice aproximado
escrito desde cero. Este es un comportamiento real y bien documentado de
los índices ANN -- un grafo NSW de una sola capa hecho a mano tiene una
sobrecarga por paso notablemente mayor (operaciones con heaps, acceso
aleatorio a memoria a través del grafo, contabilidad del conjunto de nodos
visitados) que la que disfruta un escaneo lineal amigable con la caché, y
el HNSW multicapa (el enfoque de Faiss/hnswlib, aún no implementado aquí --
ver el roadmap) existe precisamente para ampliar esa brecha a mayor
escala. La conclusión honesta: el índice ANN de este proyecto demuestra
correctamente la *estructura de datos y el algoritmo* (ver el test de
regresión de recall en `tests/test_index.py`), y el punto de cruce en el
que se amortiza depende del número de núcleos, de `ef` y de cuán
agrupados estén tus embeddings reales -- no es una afirmación universal
de "siempre más rápido", y este README no va a pretender que lo sea.

## Pruebas

```bash
pip install -e ".[dev]"
pytest --cov                 # 71 tests: calculator safety, chunking,
                              # native-vs-Python index parity, NSW recall,
                              # agent loop, RAG store, FastAPI endpoints
ruff check . && ruff format --check .
mypy -p cognivore
```

Todo lo anterior es exactamente lo que ejecuta la CI
(`.github/workflows/ci.yml`), en Ubuntu/macOS/Windows y Python 3.10-3.12;
los tests que dependen únicamente de la extensión nativa se omiten a sí
mismos (en lugar de fallar) en la variante de la matriz donde no está
disponible la cadena de herramientas de C++, imitando el mismo
comportamiento de fallback que tiene el propio paquete.

## Estructura del proyecto

Ver [docs/architecture.md](../architecture.md) (en inglés) para un
diagrama y el razonamiento detrás de las dos decisiones de diseño más
importantes (un bucle ReAct basado en texto en lugar de function calling
específico de cada proveedor, y un índice escrito a mano en lugar de
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

## Licencia

MIT – consulta [LICENSE](../../LICENSE).

## Contacto

Creado por **Sergeev Anton** ([GitHub](https://github.com/Anton-Sergeev-EA), [avsergeev1981@gmail.com](mailto:avsergeev1981@gmail.com)). Las contribuciones son bienvenidas – consulta [CONTRIBUTING.md](../../CONTRIBUTING.md).

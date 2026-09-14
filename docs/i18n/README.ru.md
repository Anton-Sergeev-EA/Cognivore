# Cognivore

**Локальный (local-first) мультимодальный агентный фреймворк без PyTorch.**
Рассуждения LLM + RAG, с написанным на C++ векторным индексом и веб-чатом —
всё работает на обычном CPU-ноутбуке, ничего не уходит за пределы вашей
машины.

[![CI](https://github.com/Anton-Sergeev-EA/cognivore/actions/workflows/ci.yml/badge.svg)](https://github.com/Anton-Sergeev-EA/cognivore/actions/workflows/ci.yml)
[![CodeQL](https://github.com/Anton-Sergeev-EA/cognivore/actions/workflows/codeql.yml/badge.svg)](https://github.com/Anton-Sergeev-EA/cognivore/actions/workflows/codeql.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](../../LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](../../pyproject.toml)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-black)](https://github.com/astral-sh/ruff)

🌐 **Читать на другом языке:** [English](../../README.md) | [Русский](README.ru.md) | [Deutsch](README.de.md) | [Français](README.fr.md) | [Italiano](README.it.md) | [Español](README.es.md) | [中文](README.zh.md) | [日本語](README.ja.md) | [हिन्दी](README.hi.md)

Cognivore — это агентный фреймворк в стиле ReAct с retrieval-augmented
generation и поддержкой мультимодальных инструментов (распознавание речи,
анализ сцен в видео), спроектированный специально для работы целиком на
CPU, без единой зависимости от PyTorch во всём стеке. Его поисковый индекс —
написанное с нуля ядро на C++ (AVX2 SIMD, параллельный точный поиск через
OpenMP и приближённый граф NSW), доступное из Python через pybind11, с
резервным вариантом на чистом NumPy, чтобы `pip install` никогда не падал
из-за отсутствия компилятора C++.

```
you> What does the ingested spec say about the retry policy, and what's 15% of that timeout in ms?

  [tool] search_knowledge_base({"query": "retry policy timeout"})
  [tool] calculator({"expression": "5000 * 0.15"})

cognivore> The spec (spec.md) sets a 5000ms timeout with exponential
backoff. 15% of that is 750ms.
```

Тот же цикл — в веб-интерфейсе: трассировка вызова инструмента и найденного фрагмента, на любом из 9 языков и в любой из 3 тем:

| Тема aurora, английский — трассировка RAG | Тёмная тема, русский — вызов калькулятора |
|---|---|
| ![Веб-интерфейс Cognivore: тема aurora, английский, вызов инструмента search_knowledge_base и найденный фрагмент](../screenshots/web-ui-en.png) | ![Веб-интерфейс Cognivore: тёмная тема, русский, вызов инструмента calculator](../screenshots/web-ui-ru.png) |

## Зачем этот проект существует

Большинство «AI-агентных» портфолио-проектов — это тонкая обёртка над
вызовом API. Этот проект показывает обратное: фреймворк, где интересные
части (настоящая структура данных ANN, провайдер-независимый протокол
вызова инструментов, гибридный конвейер поиска, аккуратная деградация при
отсутствии опциональных зависимостей) реализованы самостоятельно, а не
просто импортированы из готовой библиотеки.

## Возможности

- **Цикл агента ReAct** (Thought → Action → Observation), работающий с
  *любой* инструктивно обученной локальной моделью, а не только с
  моделями, файнтюненными под конкретный формат function calling — см.
  [docs/architecture.md](../architecture.md#why-a-react-loop-instead-of-native-function-calling).
- **Нативный векторный индекс на C++** (`native/vector_index.cpp`):
  точный `FlatIndex` (скалярное произведение AVX2/FMA, параллельный
  скан через OpenMP) и приближённый `NSWIndex` (написанный с нуля
  однослойный граф Navigable Small World — вставка, отсечение соседей,
  жадный поиск луча), оба привязаны к Python через написанное вручную
  расширение pybind11 со stub-файлом `.pyi`. Автоматически переключается
  на чистый NumPy-индекс, если во время установки нет компилятора C++.
- **Гибридный RAG**: чанкинг через рекурсивный сплиттер, семантические
  эмбеддинги `fastembed` (ONNX, без PyTorch) с резервным вариантом на
  hashing-trick без загрузок, гибридный поиск (вектор + BM25).
- **Мультимодальные инструменты**: безопасный (на основе AST, без
  `eval`) калькулятор, поиск по базе знаний, распознавание речи с грубым
  разделением по говорящим (`faster-whisper`) и анализ сцен видео
  (OpenCV) + распознавание текста на кадрах OCR (Tesseract) — всё на CPU,
  без PyTorch.
- **Подключаемый LLM-backend**: локальный инференс GGUF через
  `llama-cpp-python`, либо детерминированный `FakeLLMBackend` без единой
  зависимости, который проходит тот же самый код вызова инструментов без
  единой загрузки — именно на нём работают тесты и CI.
- **Веб-интерфейс**: FastAPI + потоковая передача через SSE + чат-интерфейс
  на чистом JS/HTML/CSS (без сборки, без фреймворка) с drag-and-drop для
  файлов/аудио/видео, живыми индикаторами «думает»/соединения, темами
  тёмная/светлая/aurora и локализацией на английский, русский, немецкий,
  французский, итальянский, испанский, упрощённый китайский, японский и
  хинди.
- Полный набор тестов, линтер+форматирование ruff, mypy (в строгом режиме,
  включая stub для нативного расширения) и CI-матрица для нескольких
  ОС/версий Python.

## Быстрый старт

```bash
git clone https://github.com/Anton-Sergeev-EA/cognivore.git
cd cognivore
python3 -m venv .venv && source .venv/bin/activate
pip install -e .                 # собирает нативное расширение, если есть компилятор C++17

cognivore chat                   # интерактивный чат, по умолчанию офлайн-демо-режим
cognivore ingest ./docs          # индексирует папку с markdown/текстовыми файлами
cognivore serve                  # веб-чат на http://127.0.0.1:8420
```

По умолчанию никакая LLM не загружается и сети не требуется вообще: агент
работает на `FakeLLMBackend` — небольшом детерминированном backend'е,
который всё равно проходит реальный цикл вызова инструментов (попробуйте
`What is 12 * 7?` или `search the knowledge base for ...` после
`ingest`). Чтобы подключить настоящую локальную LLM:

```bash
pip install -e ".[llm]"
# скачайте, например, https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF
cp .env.example .env
echo 'COGNIVORE_LLM_MODEL_PATH=/path/to/model.gguf' >> .env
cognivore chat
```

Либо проще, без компилятора C++ вообще — подключите
[Ollama](https://ollama.com): `COGNIVORE_LLM_PROVIDER=auto` (значение по
умолчанию) сначала пробует локально запущенный сервер Ollama, и только
потом падает обратно на путь к GGUF-файлу или `FakeLLMBackend`:

```bash
ollama pull qwen2.5:3b   # подходит любая инструктивно обученная модель
cognivore chat            # автоматически подхватит запущенный сервер Ollama
```

Аудио/видео-инструменты требуют отдельных extras:
`pip install -e ".[audio,video]"` (или `.[all]` для всего сразу, включая
GGUF). Полный список настроек — в `.env.example`.

Для распознавания текста на кадрах видео (`analyze_video`) дополнительно
нужен сам бинарник [Tesseract](https://github.com/tesseract-ocr/tesseract)
— пакет `pytesseract`, который ставится вместе с extra `video`, это лишь
тонкая обёртка вокруг него, и без бинарника OCR молча возвращает пустой
текст (детекция сцен и таймкоды при этом всё равно работают, поскольку
это чистый OpenCV):

```bash
sudo apt install tesseract-ocr        # Debian/Ubuntu
brew install tesseract                # macOS
# Windows: https://github.com/UB-Mannheim/tesseract/wiki
```

В Docker-образе он уже есть — устанавливать ничего не нужно.

### Docker

Образ собирается в несколько стадий (компилирует нативное расширение,
затем отбрасывает компилятор для компактного runtime-образа), запускается
не от root, и снабжён `HEALTHCHECK` — собирается один раз с кешированием
слоёв в GitHub Actions и проверяется целиком (сервер реально отвечает на
`/api/health`, контейнер сообщает `healthy`, работает не от root) при
каждом push — то есть не просто «собралось», а «проверено».

**Самый простой способ — без Python, без установки Ollama, одинаково
работает на Windows, macOS и Linux:**

```bash
docker compose up -d --build
docker compose exec ollama ollama pull qwen2.5:3b   # разово, ~2 ГБ
```

Затем откройте <http://127.0.0.1:8420>. `docker-compose.yml` запускает
Ollama *тоже* в отдельном контейнере, так что на хосте не нужно ничего
устанавливать кроме самого Docker; Cognivore обращается к нему внутри
docker-сети по имени сервиса (`http://ollama:11434`), что полностью
избегает разницы в сетевых настройках между Windows/macOS/Linux. Без
разового `ollama pull` Cognivore всё равно запустится и заработает — просто
откатится в офлайн-демо-режим `FakeLLMBackend`, пока модель не появится.

Кроме того, стек compose устанавливает `COGNIVORE_SEED_DEMO_KB=true`, поэтому
свежая база знаний автоматически заполняется двумя демонстрационными базами
знаний компаний (на английском и русском — тарифы, SLA, безопасность, политика
возврата, частые вопросы) вместо пустой зоны загрузки. Заполняется только
*пустая* база знаний: как только вы загрузите свои документы, это навсегда
становится no-op. Установите `false` в `docker-compose.yml` (или `docker run
-e COGNIVORE_SEED_DEMO_KB=false`), чтобы начать с пустой базы, либо запустите
`cognivore seed-demo` в любой момент, чтобы добавить те же демо-документы в
уже существующее хранилище.

**Уже есть Ollama на хосте, или нужен один контейнер?**

```bash
docker build -t cognivore .
docker run -d -p 8420:8420 -v cognivore-data:/data \
  -e COGNIVORE_OLLAMA_HOST=http://host.docker.internal:11434 \
  --add-host=host.docker.internal:host-gateway \
  cognivore
```

`host.docker.internal` предоставляется автоматически в Docker Desktop
(Windows/macOS); явный `--add-host` выше — то, что заставляет ту же самую
команду работать и на обычном Linux, где иначе это имя не резолвится.

Образ включает аудио/видео-инструменты (`faster-whisper`, OpenCV, и
Tesseract для распознавания текста на кадрах), но *не* `llama-cpp-python` — он общается с Ollama по обычному HTTP вместо
загрузки GGUF-файла прямо в процессе, намеренно, поскольку у
llama-cpp-python нет готового пакета под каждую платформу, а компилятора
в runtime-стадии нет. Всё равно хотите инференс GGUF прямо в контейнере?
Добавьте `build-essential` в финальную стадию и переключите её
`pip install` обратно на extra `[all]`.

**Готовый образ (без единого шага сборки):** релизы с тегами публикуются
мультиархитектурно (amd64 + arm64 — включая Apple Silicon и Raspberry Pi)
в GHCR через [`.github/workflows/docker-publish.yml`](../../.github/workflows/docker-publish.yml):

```bash
docker pull ghcr.io/anton-sergeev-ea/cognivore:latest
```

## Использование

**CLI** (`cognivore --help` — полный список команд):

```bash
cognivore chat                              # интерактивный REPL
cognivore ingest ./docs                     # рекурсивно индексирует .txt/.md/.markdown/.rst
cognivore seed-demo                         # добавляет демо-базы знаний на английском и русском
cognivore serve --host 0.0.0.0 --port 8420  # веб-интерфейс + REST/SSE API
cognivore bench                             # бенчмарки построения/поиска индекса (см. ниже)
```

**Веб-интерфейс** (`cognivore serve`, затем откройте выведенный адрес):
чат с живой потоковой передачей токенов, зона drag-and-drop для загрузки
`.txt`/`.md`, а также аудио/видео-файлов, переключатель языка (9 языков),
три темы (тёмная/светлая/aurora) и трассировка каждого вызова инструмента,
который агент сделал для конкретного ответа (клик по шагу показывает
полный, не обрезанный результат инструмента).

**REST/SSE API**, после запуска `cognivore serve`:

```bash
curl http://127.0.0.1:8420/api/health

curl -X POST http://127.0.0.1:8420/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is 12 * 7?"}'

curl -N "http://127.0.0.1:8420/api/chat/stream?message=Summarize+the+ingested+docs"

curl -X POST http://127.0.0.1:8420/api/ingest/file -F "file=@./notes.md"
```

**Как библиотека**, а не через CLI или API (см. `examples/`):

```python
from cognivore.bootstrap import build_agent
from cognivore.config import get_settings

agent = build_agent(get_settings())
result = agent.run("What's 15% of 5000?")
print(result.answer)
```

## Бенчмарки

Цифры из `python benchmarks/bench_index.py` на 2-ядерной машине
CI-класса (образ `Dockerfile` переносим; на вашем реальном ноутбуке почти
наверняка больше ядер, и это важно — см. ниже). Случайные,
равномерно распределённые 384-мерные векторы — это близко к *худшему
случаю* для приближённого поиска (нет реальной кластерной структуры,
поэтому «ближайшие» соседи лишь незначительно ближе случайных); на
реальных эмбеддингах recall заметно выше при том же `ef`.

**Время построения** (3 000 векторов) — здесь преимущество нативного
расширения однозначно:

| Индекс | Время построения | vs. NumPy-резерв |
|---|---|---|
| `FlatIndexPy` (NumPy, `.add()` в цикле) | 0.615с | 1x |
| `FlatIndex` (C++) | 0.027с | **~в 23 раза быстрее** |

**Recall и скорость — это настраиваемый параметр, а не одно число**
(`NSWIndex`, n=3 000):

| `ef` | мс/запрос | Recall@10 |
|---|---|---|
| 50  | 0.24 | 0.64 |
| 150 (по умолчанию) | 0.46 | 0.93 |
| 400 | 0.78 | 1.00 |

**Задержка поиска в зависимости от размера коллекции** — точный линейный
скан (даже с ускорением AVX2+OpenMP) *нормален*, пока коллекция не
становится настолько большой, что сам скан становится узким местом; именно
в этой точке приближённый граф должен начинать выигрывать:

| n | `FlatIndex` мс/запрос | `NSWIndex` мс/запрос (ef=150) | Прирост |
|---|---|---|---|
| 1 000 | 0.036 | 0.240 | 0.2x (брутфорс выигрывает — коллекция слишком мала) |
| 10 000 | 0.340 | 0.798 | 0.4x |
| 50 000 | 1.561 | 1.291 | 1.2x |

Читайте эту таблицу как есть, а не так, как хотелось бы для красивой
истории: на этих размерах, на 2 ядрах, SIMD+параллельный брутфорс-скан
*конкурентен или быстрее* приближённого индекса, написанного с нуля. Это
реальное, хорошо документированное поведение ANN-индексов — у
однослойного NSW-графа, написанного вручную, заметно больше накладных
расходов на шаг (операции с кучей, произвольный доступ к памяти по графу,
учёт посещённых узлов), чем у дружелюбного к кешу линейного скана, а
многослойный HNSW (подход Faiss/hnswlib, здесь пока не реализован — см.
roadmap) существует специально, чтобы увеличить этот разрыв на больших
масштабах. Честный вывод: ANN-индекс этого проекта корректно демонстрирует
*структуру данных и алгоритм* (см. регрессионный тест recall в
`tests/test_index.py`), а точка, где он начинает окупаться, зависит от
количества ядер, `ef` и того, насколько кластеризованы ваши реальные
эмбеддинги — это не универсальное утверждение «всегда быстрее», и этот
README не будет притворяться, что это так.

## Тестирование

```bash
pip install -e ".[dev]"
pytest --cov                 # 71 тест: безопасность калькулятора, чанкинг,
                              # паритет нативного и Python-индекса, recall NSW,
                              # цикл агента, RAG-хранилище, эндпоинты FastAPI
ruff check . && ruff format --check .
mypy -p cognivore
```

Всё вышеперечисленное — это именно то, что запускает CI
(`.github/workflows/ci.yml`), на Ubuntu/macOS/Windows и Python 3.10-3.12;
тесты, зависящие только от нативного расширения, самостоятельно
пропускаются (а не падают) на той конфигурации матрицы, где недоступен
C++-тулчейн — так же, как ведёт себя сам пакет.

## Структура проекта

См. [docs/architecture.md](../architecture.md) (на английском) для схемы
и обоснования двух главных архитектурных решений (текстовый цикл ReAct
вместо function calling конкретного провайдера, и написанный вручную
индекс вместо Faiss/hnswlib).

```
native/            ядро векторного индекса на C++ + привязки pybind11
src/cognivore/
  index/            выбор нативного индекса или резервного варианта
  rag/              чанкинг, эмбеддинги, гибридное хранилище документов
  agent/            цикл ReAct, память, промпты/парсинг
  tools/            калькулятор, поиск RAG, аудио, видео
  llm/              backend llama.cpp + FakeLLMBackend
  media/            обёртки над faster-whisper / OpenCV
  web/              приложение FastAPI + статический чат-интерфейс
  cli.py            cognivore chat|ingest|serve|bench
benchmarks/         отдельные скрипты для цифр выше
examples/           минимальные скрипты использования библиотеки
tests/              набор тестов pytest (71 тест)
```

## Лицензия

MIT — см. [LICENSE](../../LICENSE).

## Контакты

Разработчик — **Sergeev Anton** ([GitHub](https://github.com/Anton-Sergeev-EA), [avsergeev1981@gmail.com](mailto:avsergeev1981@gmail.com)).
Контрибуции приветствуются — см. [CONTRIBUTING.md](../../CONTRIBUTING.md).

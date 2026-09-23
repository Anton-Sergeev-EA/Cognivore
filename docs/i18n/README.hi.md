# Cognivore

**एक लोकल-फर्स्ट, PyTorch-मुक्त मल्टीमॉडल एजेंट फ्रेमवर्क।** LLM रीज़निंग +
RAG, हाथ से लिखे गए C++ वेक्टर इंडेक्स और एक वेब चैट UI के साथ -- सब कुछ
एक CPU-only लैपटॉप पर चलता है, कुछ भी आपकी मशीन से बाहर जाने की ज़रूरत
नहीं है।

[![CI](https://github.com/Anton-Sergeev-EA/cognivore/actions/workflows/ci.yml/badge.svg)](https://github.com/Anton-Sergeev-EA/cognivore/actions/workflows/ci.yml)
[![CodeQL](https://github.com/Anton-Sergeev-EA/cognivore/actions/workflows/codeql.yml/badge.svg)](https://github.com/Anton-Sergeev-EA/cognivore/actions/workflows/codeql.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](../../LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](../../pyproject.toml)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-black)](https://github.com/astral-sh/ruff)

🌐 **इसे किसी अन्य भाषा में पढ़ें:** [English](../../README.md) | [Русский](README.ru.md) | [Deutsch](README.de.md) | [Français](README.fr.md) | [Italiano](README.it.md) | [Español](README.es.md) | [中文](README.zh.md) | [日本語](README.ja.md) | [हिन्दी](README.hi.md)

Cognivore एक ReAct-स्टाइल एजेंट फ्रेमवर्क है जिसमें retrieval-augmented
generation और मल्टीमॉडल टूल सपोर्ट (ऑडियो ट्रांसक्रिप्शन, वीडियो सीन
एनालिसिस) शामिल है, और इसे खास तौर पर पूरी तरह से CPU-only मशीन पर चलाने
के लिए बनाया गया है, पूरे स्टैक में कहीं भी PyTorch पर कोई निर्भरता नहीं
है। इसका रिट्रीवल इंडेक्स एक हाथ से लिखा गया C++ कोर है (AVX2 SIMD,
OpenMP-parallel exact search, और शुरू से लिखा गया approximate NSW graph),
जो pybind11 के ज़रिए Python को एक्सपोज़ किया गया है, साथ ही एक pure-NumPy
फॉलबैक भी है, ताकि C++ कंपाइलर न होने की वजह से `pip install` कभी fail न
हो।

```
you> What does the ingested spec say about the retry policy, and what's 15% of that timeout in ms?

  [tool] search_knowledge_base({"query": "retry policy timeout"})
  [tool] calculator({"expression": "5000 * 0.15"})

cognivore> The spec (spec.md) sets a 5000ms timeout with exponential
backoff. 15% of that is 750ms.
```

वही लूप, वेब UI में — टूल कॉल और उसके द्वारा प्राप्त पैसेज का लाइव ट्रेस, सभी 9 भाषाओं और 3 थीम में से किसी में भी:

| Aurora थीम, अंग्रेज़ी — RAG ट्रेस | डार्क थीम, रूसी — कैलकुलेटर |
|---|---|
| ![Cognivore वेब UI: Aurora थीम, अंग्रेज़ी, एक search_knowledge_base टूल कॉल और प्राप्त पैसेज](../screenshots/web-ui-en.png) | ![Cognivore वेब UI: डार्क थीम, रूसी, एक calculator टूल कॉल](../screenshots/web-ui-ru.png) |

## यह प्रोजेक्ट क्यों बनाया गया

ज़्यादातर "AI एजेंट" पोर्टफोलियो प्रोजेक्ट्स किसी API कॉल के ऊपर एक पतली
रैपर होते हैं। यह प्रोजेक्ट इसके उलट दिखाने के लिए बनाया गया है: एक ऐसा
फ्रेमवर्क जिसमें दिलचस्प हिस्से (एक असली ANN डेटा स्ट्रक्चर, एक
provider-agnostic टूल-कॉलिंग प्रोटोकॉल, एक हाइब्रिड रिट्रीवल पाइपलाइन,
ऑप्शनल डिपेंडेंसीज़ के गायब होने पर सहज gracefully degradation) खुद
इम्प्लीमेंट किए गए हैं, न कि किसी लाइब्रेरी से इम्पोर्ट किए गए हैं।

## विशेषताएँ

- **ReAct एजेंट लूप** (Thought → Action → Observation) जो *किसी भी*
  इंस्ट्रक्शन-ट्यून्ड लोकल मॉडल के साथ काम करता है, न कि सिर्फ किसी खास
  function-calling वायर फॉर्मैट के लिए फाइनट्यून्ड मॉडलों के साथ -- देखें
  [docs/architecture.md](../architecture.md#why-a-react-loop-instead-of-native-function-calling)।
- **नेटिव C++ वेक्टर इंडेक्स** (`native/vector_index.cpp`): एक exact
  `FlatIndex` (AVX2/FMA डॉट प्रोडक्ट, OpenMP-parallel scan) और एक
  approximate `NSWIndex` (शुरू से लिखा गया single-layer Navigable Small
  World graph -- insertion, neighbour pruning, greedy beam search), दोनों
  को Python से हाथ से लिखे गए pybind11 एक्सटेंशन के ज़रिए `.pyi` type
  stub के साथ बाइंड किया गया है। इंस्टॉल के समय C++ कंपाइलर न होने पर यह
  ऑटोमैटिकली pure-NumPy इंडेक्स पर फॉलबैक कर जाता है।
- **हाइब्रिड RAG**: recursive-splitter chunking, `fastembed` (ONNX, बिना
  PyTorch) सिमैंटिक एम्बेडिंग्स एक zero-download hashing-trick फॉलबैक के
  साथ, वेक्टर + BM25 हाइब्रिड रिट्रीवल।
- **मल्टीमॉडल टूल्स**: सुरक्षित (AST-आधारित, बिना `eval`) कैलकुलेटर,
  नॉलेज-बेस सर्च, ऑडियो ट्रांसक्रिप्शन + मोटे तौर पर speaker turns
  (`faster-whisper`), और वीडियो scene-detection (OpenCV) + ऑन-स्क्रीन टेक्स्ट OCR
  (Tesseract) -- सब कुछ
  CPU-only, बिना PyTorch।
- **प्लगेबल LLM बैकएंड**: `llama-cpp-python` के ज़रिए लोकल GGUF इनफेरेंस,
  या एक deterministic, dependency-free `FakeLLMBackend` जो बिना कुछ भी
  डाउनलोड किए वही टूल-कॉलिंग कोड पाथ चलाता है -- यही वह है जिसके ऊपर टेस्ट
  सूट और CI चलते हैं।
- **Web UI**: FastAPI + SSE स्ट्रीमिंग + एक vanilla JS/HTML/CSS चैट
  इंटरफ़ेस (कोई बिल्ड स्टेप नहीं, कोई फ्रेमवर्क नहीं) फाइल/ऑडियो/वीडियो
  के लिए drag-and-drop के साथ, live "thinking"/connection इंडिकेटर्स,
  dark/light/aurora थीम्स, और अंग्रेज़ी, रूसी, जर्मन, फ्रेंच, इतालवी,
  स्पैनिश, सरलीकृत चीनी, जापानी, और हिन्दी में लोकलाइज़ेशन।
- पूरा टेस्ट सूट, ruff lint+format, mypy (लगभग strict मोड में, नेटिव
  एक्सटेंशन के लिए एक stub शामिल), और कई OS/कई Python वर्ज़न वाली CI
  मैट्रिक्स।

## क्विकस्टार्ट

```bash
git clone https://github.com/Anton-Sergeev-EA/cognivore.git
cd cognivore
python3 -m venv .venv && source .venv/bin/activate
pip install -e .                 # builds the native extension if a C++17 compiler is present

cognivore chat                   # interactive chat, offline demo mode by default
cognivore ingest ./docs          # index a folder of markdown/text files
cognivore serve                  # web chat UI at http://127.0.0.1:8420
```

डिफ़ॉल्ट रूप से कोई LLM डाउनलोड नहीं होता और बिल्कुल कोई नेटवर्क एक्सेस
नहीं होती: एजेंट `FakeLLMBackend` के ऊपर चलता है, जो एक छोटा
deterministic बैकएंड है जो फिर भी असली टूल-कॉलिंग लूप चलाता है (कुछ
`ingest` करने के बाद `What is 12 * 7?` या `search the knowledge base
for ...` ट्राई करें)। किसी असली लोकल LLM का इस्तेमाल करने के लिए:

```bash
pip install -e ".[llm]"
# download e.g. https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF
cp .env.example .env
echo 'COGNIVORE_LLM_MODEL_PATH=/path/to/model.gguf' >> .env
cognivore chat
```

या, ज़्यादा आसान तरीका और बिना किसी C++ टूलचेन के, इसे इसके बजाय
[Ollama](https://ollama.com) की ओर पॉइंट करें -- `COGNIVORE_LLM_PROVIDER=auto`
(डिफ़ॉल्ट) पहले लोकली चल रहे Ollama सर्वर को ट्राई करता है, फिर GGUF
पाथ या `FakeLLMBackend` पर फॉलबैक करता है:

```bash
ollama pull qwen2.5:3b   # any instruction-tuned model works
cognivore chat           # picks up the running Ollama server automatically
```

ऑडियो/वीडियो टूल्स को अपने खुद के extras चाहिए होते हैं:
`pip install -e ".[audio,video]"` (या सब कुछ के लिए `.[all]`, GGUF
शामिल)। हर सेटिंग के लिए `.env.example` देखें।

वीडियो टूल (`analyze_video`) में ऑन-स्क्रीन टेक्स्ट एक्सट्रैक्शन के लिए
[Tesseract](https://github.com/tesseract-ocr/tesseract) OCR *बाइनरी* भी
चाहिए होती है -- `video` extra के साथ आने वाला `pytesseract` पैकेज तो बस
इसके ऊपर एक पतला wrapper है, और इसके बिना OCR बिना किसी चेतावनी के कोई
टेक्स्ट नहीं लौटाता (scene detection और timestamps दोनों तरीकों से चलते
रहते हैं, क्योंकि वह हिस्सा शुद्ध रूप से OpenCV पर आधारित है)। डिफ़ॉल्ट रूप
से यह English *और* Russian दोनों को पहचानता है (`eng+rus`, `.env.example`
में `COGNIVORE_OCR_LANGUAGES` देखें) -- Debian/Ubuntu पर plain
`tesseract-ocr` पैकेज `eng` को खुद-ब-खुद ले आता है, लेकिन `rus` को नहीं,
इसलिए दोनों को अलग से इंस्टॉल करें:

```bash
sudo apt install tesseract-ocr tesseract-ocr-rus   # Debian/Ubuntu
brew install tesseract                              # macOS -- सभी भाषाएँ साथ ही आती हैं
# Windows: https://github.com/UB-Mannheim/tesseract/wiki (इंस्टॉलर की language list में Russian को टिक करें)
```

Docker इमेज में दोनों पहले से ही शामिल हैं -- वहाँ कुछ भी इंस्टॉल करने की
ज़रूरत नहीं।

जिस भाषा का trained-data पैकेज इंस्टॉल न हो, उसे रिक्वेस्ट करने पर एरर
नहीं आता -- वह स्क्रिप्ट को चुपचाप मिलती-जुलती Latin letters में गलत
पहचान लेता है (Cyrillic "Контейнеры" "KoHTewHepbi" के रूप में सामने आता
है), जो missing language pack जैसा नहीं बल्कि एक खराब scan जैसा दिखता है।
अगर आपके ऑन-स्क्रीन टेक्स्ट में कोई और भाषा है, तो उसका
`tesseract-ocr-<lang>` पैकेज इंस्टॉल करें और उसे `COGNIVORE_OCR_LANGUAGES`
में जोड़ें (जैसे `eng+rus+deu`)।

`analyze_video` का OCR उस तरह के कंटेंट के लिए बनाया गया है जिसका ज़िक्र
इस tool के अपने description में ही है -- screencasts, lecture recordings,
slide-based videos -- जहाँ टेक्स्ट बड़ा और सोच-समझकर composed होता है।
किसी raw terminal/IDE स्क्रीन रिकॉर्डिंग पर यह कहीं ज़्यादा rough साबित
होता है: छोटा monospace font, ठीक उन scene cuts पर भारी video
compression जिन्हें यह tool अपना आधार बनाता है, और box-drawing या symbol
glyphs जिन पर OCR मॉडल कभी trained ही नहीं हुए। OCR से पहले frame को
upscale या threshold करना भरोसे के साथ मदद नहीं करता, जब compression
पहले ही fine detail गँवा चुका हो -- यह testing से confirm किया गया है,
सिर्फ़ अनुमान नहीं। अगर आप specifically चाहते हैं कि on-screen
terminal/code टेक्स्ट अच्छे से OCR हो, तो बड़े font size और/या ज़्यादा
resolution पर रिकॉर्ड करें; यही असली lever है जो काम करता है, बाद में
post-processing नहीं।

### Docker

यह इमेज एक multi-stage build है (नेटिव C++ एक्सटेंशन को कंपाइल करती है,
फिर एक slim runtime इमेज के लिए कंपाइलर को हटा देती है), non-root यूज़र
के तौर पर चलती है, और एक `HEALTHCHECK` के साथ आती है -- यह GitHub
Actions की Docker layer caching के साथ एक बार बिल्ड होती है और हर push
पर end-to-end वेरिफाई की जाती है (सर्वर वास्तव में `/api/health` का
जवाब देता है, कंटेनर `healthy` रिपोर्ट करता है, non-root के तौर पर चलता
है), यानी यह सिर्फ "बिल्ड होती है" नहीं है, बल्कि "चेक की गई" है।

**सबसे आसान तरीका -- न Python चाहिए, न Ollama इंस्टॉल, Windows, macOS,
और Linux पर एक जैसा काम करता है:**

```bash
docker compose up -d --build
docker compose exec ollama ollama pull qwen2.5:3b   # one-time, ~2GB
```

फिर <http://127.0.0.1:8420> खोलें। `docker-compose.yml` Ollama को भी
*अपने* कंटेनर के अंदर चलाता है, इसलिए host पर Docker के अलावा कुछ भी
इंस्टॉल करने की ज़रूरत नहीं है; Cognivore compose नेटवर्क पर सर्विस के
नाम से (`http://ollama:11434`) उस तक पहुँचता है, जो Windows/macOS/Linux
के बीच होने वाले सामान्य host-networking अंतरों को पूरी तरह से टाल देता
है। बिना उस one-time `ollama pull` के भी, Cognivore फिर भी शुरू होता है
और ठीक से चलता है -- यह सिर्फ तब तक ऑफ़लाइन `FakeLLMBackend` डेमो मोड पर
फॉलबैक करता है जब तक कोई मॉडल उपलब्ध नहीं हो जाता।

compose स्टैक `COGNIVORE_SEED_DEMO_KB=true` भी सेट करता है, इसलिए एक नई (खाली)
knowledge base खाली dropzone दिखाने के बजाय दो बंडल किए गए डेमो कंपनी हैंडबुक
(अंग्रेज़ी + रूसी — प्राइसिंग, SLA, सुरक्षा, रिफंड पॉलिसी, सपोर्ट FAQ) से
अपने-आप भर जाती है। यह हमेशा केवल *खाली* knowledge base को ही भरता है: जैसे ही
आप अपने खुद के दस्तावेज़ ingest करते हैं, यह स्थायी रूप से no-op बन जाता है।
खाली शुरू करने के लिए `docker-compose.yml` में इसे `false` सेट करें (या
`docker run -e COGNIVORE_SEED_DEMO_KB=false`), या किसी भी समय `cognivore
seed-demo` चलाकर उन्हीं डेमो दस्तावेज़ों को किसी मौजूदा store में जोड़ें।

**पहले से host पर Ollama चल रहा है, या एक ही कंटेनर चाहिए?**

```bash
docker build -t cognivore .
docker run -d -p 8420:8420 -v cognivore-data:/data \
  -e COGNIVORE_OLLAMA_HOST=http://host.docker.internal:11434 \
  --add-host=host.docker.internal:host-gateway \
  cognivore
```

`host.docker.internal` Docker Desktop (Windows/macOS) पर ऑटोमैटिकली
मिलता है; ऊपर दिया गया explicit `--add-host` वही है जो इसी कमांड को
plain Linux पर भी काम करने देता है, जहाँ अन्यथा यह resolve नहीं होता।

यह इमेज ऑडियो/वीडियो टूल्स (`faster-whisper`, OpenCV, और ऑन-स्क्रीन टेक्स्ट
OCR के लिए Tesseract) के साथ आती है
लेकिन `llama-cpp-python` के *बिना* -- यह LLM के लिए in-process GGUF
फाइल लोड करने के बजाय plain HTTP पर Ollama से बात करती है, जानबूझकर,
क्योंकि llama-cpp-python के पास हर प्लेटफ़ॉर्म के लिए prebuilt wheel
नहीं है और इसे एक कंपाइलर चाहिए जो runtime स्टेज में मौजूद नहीं है। फिर
भी कंटेनर के अंदर in-process GGUF इनफेरेंस चाहिए? फाइनल स्टेज में
`build-essential` जोड़ें और उसके `pip install` को वापस `[all]` extra
पर स्विच करें।

`faster-whisper` अपना स्पीच-रिकग्निशन मॉडल Hugging Face से पहली बार
डाउनलोड करता है जब ऑडियो ट्रांसक्रिप्शन असल में इस्तेमाल होता है (build
टाइम पर नहीं) -- ठीक वैसे ही जैसे LLM के लिए `ollama pull` होता है,
फ़र्क़ बस यह है कि यह पहले इस्तेमाल पर ऑटोमैटिकली हो जाता है, कोई explicit
कमांड नहीं चाहिए। Ollama के मॉडल्स की तरह ही, यह भी persisted
`cognivore-data` volume में cache होता है (`HF_HOME=/data/hf-cache`),
इसलिए यह सिर्फ़ एक बार डाउनलोड होता है, हर `docker compose up --build`
पर नहीं।

**रोकना और फिर से शुरू करना** (जैसे reboot के बाद):

```bash
docker compose down   # दोनों कंटेनर्स को रोकता है; डेटा बना रहता है (नीचे देखें)
docker compose up -d  # फिर से शुरू करता है -- --build की ज़रूरत नहीं, जब तक इमेज खुद न बदली हो
```

दोनों services `restart: unless-stopped` पर सेट हैं, इसलिए अगर आपने shutdown से
पहले उन्हें मैन्युअली नहीं रोका, तो Docker daemon वापस ऊपर आने पर (ज़्यादातर
installs पर डिफ़ॉल्ट) उन्हें अपने-आप restart कर देता है -- उस स्थिति में किसी
कमांड की ज़रूरत ही नहीं। नॉलेज बेस और cached Whisper/Ollama मॉडल्स named
volumes `cognivore-data` और `ollama-data` में रहते हैं, जिन्हें
`docker compose down` कभी नहीं छूता; इन्हें सिर्फ़ एक explicit
`docker compose down -v` ही हटाता है।

**Prebuilt इमेज (बिल्कुल कोई बिल्ड स्टेप नहीं):** tagged रिलीज़
[`.github/workflows/docker-publish.yml`](../../.github/workflows/docker-publish.yml)
के ज़रिए multi-arch (amd64 + arm64 -- Apple Silicon और Raspberry Pi
शामिल) तरीके से GHCR पर पब्लिश होते हैं:

```bash
docker pull ghcr.io/anton-sergeev-ea/cognivore:latest
```

## उपयोग

**CLI** (पूरी लिस्ट के लिए `cognivore --help`):

```bash
cognivore chat                              # interactive REPL
cognivore ingest ./docs                     # recursively indexes .txt/.md/.markdown/.rst
cognivore seed-demo                         # बंडल किए गए EN+RU डेमो कंपनी हैंडबुक जोड़ता है
cognivore serve --host 0.0.0.0 --port 8420  # web UI + REST/SSE API
cognivore bench                             # index build/search benchmarks (see below)
```

**Web UI** (`cognivore serve`, फिर प्रिंट हुए URL को खोलें): live token
स्ट्रीमिंग के साथ एक चैट इंटरफ़ेस, `.txt`/`.md` ingestion और
ऑडियो/वीडियो फाइलों के लिए एक drag-and-drop zone, एक भाषा स्विचर (9
भाषाएँ), तीन थीम्स (dark/light/aurora), और किसी दिए गए जवाब के लिए
एजेंट द्वारा किए गए हर टूल कॉल का एक trace view (किसी स्टेप पर क्लिक
करने से पूरा, untruncated टूल आउटपुट दिखता है)।

**REST/SSE API**, `cognivore serve` चलने के बाद:

```bash
curl http://127.0.0.1:8420/api/health

curl -X POST http://127.0.0.1:8420/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is 12 * 7?"}'

curl -N "http://127.0.0.1:8420/api/chat/stream?message=Summarize+the+ingested+docs"

curl -X POST http://127.0.0.1:8420/api/ingest/file -F "file=@./notes.md"
```

**एक लाइब्रेरी के रूप में**, CLI या API के बजाय (देखें `examples/`):

```python
from cognivore.bootstrap import build_agent
from cognivore.config import get_settings

agent = build_agent(get_settings())
result = agent.run("What's 15% of 5000?")
print(result.answer)
```

## बेंचमार्क

ये नंबर एक 2-कोर CI-क्लास मशीन पर `python benchmarks/bench_index.py`
से आए हैं (`Dockerfile` की इमेज पोर्टेबल है; आपके असली लैपटॉप में
लगभग निश्चित रूप से ज़्यादा कोर हैं, जो मैटर करता है -- नीचे देखें)।
रैंडम, uniformly-distributed 384-डाइमेंशनल वेक्टर, जो approximate
सर्च के लिए *worst case* के करीब है (कोई असली क्लस्टर स्ट्रक्चर नहीं
है, इसलिए "nearest" neighbours रैंडम neighbours से सिर्फ मामूली रूप से
ज़्यादा नज़दीक होते हैं); असली एम्बेडिंग्स पर, उसी `ef` पर recall
काफ़ी बेहतर होता है।

**Build time** (3,000 वेक्टर) -- यहाँ नेटिव एक्सटेंशन का फ़ायदा साफ़
तौर पर बेशक है:

| इंडेक्स | Build time | NumPy फॉलबैक की तुलना में |
|---|---|---|
| `FlatIndexPy` (NumPy, लूप में `.add()`) | 0.615s | 1x |
| `FlatIndex` (C++) | 0.027s | **~23x तेज़** |

**Recall बनाम स्पीड एक ट्यून करने लायक नॉब है, एक ही नंबर नहीं**
(`NSWIndex`, n=3,000):

| `ef` | ms/query | Recall@10 |
|---|---|---|
| 50  | 0.24 | 0.64 |
| 150 (डिफ़ॉल्ट) | 0.46 | 0.93 |
| 400 | 0.78 | 1.00 |

**Search latency बनाम collection का साइज़** -- एक exact linear scan
(भले ही AVX2+OpenMP से accelerated हो) *ठीक* रहता है जब तक कि collection
इतना बड़ा न हो जाए कि उसे स्कैन करना ही bottleneck बन जाए; यही वह
crossover पॉइंट है जहाँ एक approximate graph इंडेक्स को जीतना शुरू करना
चाहिए:

| n | `FlatIndex` ms/query | `NSWIndex` ms/query (ef=150) | Speedup |
|---|---|---|---|
| 1,000 | 0.036 | 0.240 | 0.2x (brute force जीतता है -- collection बहुत छोटा है) |
| 10,000 | 0.340 | 0.798 | 0.4x |
| 50,000 | 1.561 | 1.291 | 1.2x |

इस टेबल को वैसे पढ़ें जैसा यह वास्तव में कहती है, न कि वैसा जो एक बेहतर
कहानी बनाए: इन साइज़ों पर, 2 कोर पर, एक SIMD+parallel brute-force scan
शुरू से लिखे गए approximate इंडेक्स के *मुक़ाबले में है या उससे तेज़*
है। यह असली, अच्छी तरह डॉक्युमेंटेड ANN behavior है -- हाथ से लिखे गए
single-layer NSW graph में per-step overhead (heap operations,
ग्राफ़ में random memory access, visited-set bookkeeping) कैश-फ्रेंडली
लीनियर स्कैन के मुक़ाबले काफ़ी ज़्यादा है, और multi-layer HNSW
(Faiss/hnswlib का अप्रोच, यहाँ अभी इम्प्लीमेंट नहीं किया गया -- roadmap
देखें) खास तौर पर इस अंतर को स्केल पर बढ़ाने के लिए मौजूद है। ईमानदार
नतीजा यह है: इस प्रोजेक्ट का ANN इंडेक्स *डेटा स्ट्रक्चर और एल्गोरिदम*
को सही तरीके से दिखाता है (देखें `tests/test_index.py` का recall
regression टेस्ट), और वह crossover पॉइंट जहाँ यह खुद के लिए भुगतान
करता है, कोर काउंट, `ef`, और आपके असली एम्बेडिंग्स कितने क्लस्टर्ड हैं
इस पर निर्भर करके बदलता है -- यह कोई यूनिवर्सल "हमेशा तेज़" वाला दावा
नहीं है, और यह README ऐसा होने का दिखावा नहीं करेगा।

## टेस्टिंग

```bash
pip install -e ".[dev]"
pytest --cov                 # 71 tests: calculator safety, chunking,
                              # native-vs-Python index parity, NSW recall,
                              # agent loop, RAG store, FastAPI endpoints
ruff check . && ruff format --check .
mypy -p cognivore
```

ऊपर दिया गया हर चेक वही है जो CI चलाता है (`.github/workflows/ci.yml`),
Ubuntu/macOS/Windows और Python 3.10-3.12 पर; सिर्फ नेटिव-एक्सटेंशन पर
निर्भर टेस्ट खुद को उस matrix leg पर fail करने के बजाय skip कर लेते हैं
जहाँ C++ टूलचेन उपलब्ध नहीं है, जो पैकेज के अपने फॉलबैक बिहेवियर से
मेल खाता है।

## प्रोजेक्ट संरचना

डायग्राम और दो सबसे बड़े डिज़ाइन फैसलों (provider-specific function
calling के बजाय एक टेक्स्ट-आधारित ReAct लूप, और Faiss/hnswlib के बजाय
हाथ से लिखा गया इंडेक्स) के पीछे की सोच के लिए
[docs/architecture.md](../architecture.md) (अंग्रेज़ी में) देखें।

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

## लाइसेंस

MIT — देखें [LICENSE](../../LICENSE)।

## संपर्क

इसे **Sergeev Anton** द्वारा बनाया गया है ([GitHub](https://github.com/Anton-Sergeev-EA), [avsergeev1981@gmail.com](mailto:avsergeev1981@gmail.com))। योगदान का स्वागत है — देखें [CONTRIBUTING.md](../../CONTRIBUTING.md)।

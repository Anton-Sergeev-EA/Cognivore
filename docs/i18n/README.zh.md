# Cognivore

**本地优先、无需 PyTorch 的多模态智能体框架。** LLM 推理 + RAG，配备手写的
C++ 向量索引与网页聊天界面 —— 一切均可在纯 CPU 笔记本电脑上运行，无需任何
内容离开你的设备。

[![CI](https://github.com/Anton-Sergeev-EA/cognivore/actions/workflows/ci.yml/badge.svg)](https://github.com/Anton-Sergeev-EA/cognivore/actions/workflows/ci.yml)
[![CodeQL](https://github.com/Anton-Sergeev-EA/cognivore/actions/workflows/codeql.yml/badge.svg)](https://github.com/Anton-Sergeev-EA/cognivore/actions/workflows/codeql.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](../../LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](../../pyproject.toml)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-black)](https://github.com/astral-sh/ruff)

🌐 **阅读其他语言版本：** [English](../../README.md) | [Русский](README.ru.md) | [Deutsch](README.de.md) | [Français](README.fr.md) | [Italiano](README.it.md) | [Español](README.es.md) | [中文](README.zh.md) | [日本語](README.ja.md) | [हिन्दी](README.hi.md)

Cognivore 是一个 ReAct 风格的智能体框架，具备检索增强生成（RAG）能力和多
模态工具支持（音频转录、视频场景分析），专为完全在纯 CPU 机器上运行而构
建，整个技术栈中不存在任何 PyTorch 依赖。它的检索索引是一个手写的 C++
核心（AVX2 SIMD、OpenMP 并行精确搜索，以及从零实现的近似 NSW 图），通过
pybind11 暴露给 Python，并配有纯 NumPy 的回退方案，因此即使没有 C++ 编译
器，`pip install` 也绝不会硬性失败。

```
you> What does the ingested spec say about the retry policy, and what's 15% of that timeout in ms?

  [tool] search_knowledge_base({"query": "retry policy timeout"})
  [tool] calculator({"expression": "5000 * 0.15"})

cognivore> The spec (spec.md) sets a 5000ms timeout with exponential
backoff. 15% of that is 750ms.
```

同样的流程，在网页界面中呈现——工具调用及其检索到的片段的实时轨迹，可在全部 9 种语言和 3 种主题中任选：

| Aurora 主题，英文——RAG 轨迹 | 深色主题，俄文——计算器 |
|---|---|
| ![Cognivore 网页界面：Aurora 主题、英文，一次 search_knowledge_base 工具调用及其检索到的片段](../screenshots/web-ui-en.png) | ![Cognivore 网页界面：深色主题、俄文，一次 calculator 工具调用](../screenshots/web-ui-ru.png) |

## 项目缘由

大多数“AI 智能体”作品集项目只是对某个 API 调用的简单封装。而这个项目要
展示的正是相反的一面：一个把有趣的部分——真实的 ANN 数据结构、与厂商无
关的工具调用协议、混合检索流水线、在缺少可选依赖时的优雅降级——都自己
实现出来，而不是直接导入现成库的框架。

## 功能特性

- **ReAct 智能体循环**（Thought → Action → Observation），可与*任意*经
  过指令微调的本地模型配合使用，而不仅限于针对特定 function-calling 传
  输格式微调过的模型 —— 详见
  [docs/architecture.md](../architecture.md#why-a-react-loop-instead-of-native-function-calling)。
- **原生 C++ 向量索引**（`native/vector_index.cpp`）：精确的 `FlatIndex`
  （AVX2/FMA 点积、OpenMP 并行扫描）和近似的 `NSWIndex`（从零实现的单
  层 Navigable Small World 图 —— 插入、邻居剪枝、贪心束搜索），两者都
  通过手写的 pybind11 扩展绑定到 Python，并附带 `.pyi` 类型存根。若安
  装时系统没有 C++ 编译器，则自动回退到纯 NumPy 索引。
- **混合 RAG**：基于递归分割器的分块（chunking）、`fastembed`（ONNX，
  无 PyTorch）语义嵌入并配有无需下载的 hashing-trick 回退方案、向量 +
  BM25 混合检索。
- **多模态工具**：安全的（基于 AST，不使用 `eval`）计算器、知识库搜索、
  音频转录 + 粗略的说话人轮次划分（`faster-whisper`），以及视频场景检
  测（OpenCV）+ 屏幕文字 OCR 识别（Tesseract）—— 全部仅需 CPU，无需
  PyTorch。
- **可插拔的 LLM 后端**：通过 `llama-cpp-python` 进行本地 GGUF 推理，
  或使用确定性、零依赖的 `FakeLLMBackend`，它无需任何下载即可走完完全
  相同的工具调用代码路径 —— 测试套件和 CI 正是针对它运行的。
- **网页界面**：FastAPI + SSE 流式传输 + 纯原生 JS/HTML/CSS 聊天界面
  （无需构建步骤，无需框架），支持文件/音频/视频拖放上传、实时“思考
  中”/连接状态指示、深色/浅色/aurora 三种主题，并本地化为英语、俄语、
  德语、法语、意大利语、西班牙语、简体中文、日语和印地语。
- 完整的测试套件、ruff 代码检查与格式化、mypy（近乎严格模式，包括针对
  原生扩展的类型存根），以及覆盖多操作系统/多 Python 版本的 CI 矩阵。

## 快速开始

```bash
git clone https://github.com/Anton-Sergeev-EA/cognivore.git
cd cognivore
python3 -m venv .venv && source .venv/bin/activate
pip install -e .                 # 如果系统中有 C++17 编译器，则会构建原生扩展

cognivore chat                   # 交互式聊天，默认为离线演示模式
cognivore ingest ./docs          # 索引一个包含 markdown/文本文件的文件夹
cognivore serve                  # 网页聊天界面，地址为 http://127.0.0.1:8420
```

默认情况下完全不需要下载任何 LLM，也不需要任何网络访问：智能体运行在
`FakeLLMBackend` 之上，这是一个小型的确定性后端，但仍会走完真实的工具调
用循环（在对某些内容执行 `ingest` 之后，可以试试输入 `What is 12 * 7?`
或 `search the knowledge base for ...`）。若要使用真正的本地 LLM：

```bash
pip install -e ".[llm]"
# 例如从 https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF 下载
cp .env.example .env
echo 'COGNIVORE_LLM_MODEL_PATH=/path/to/model.gguf' >> .env
cognivore chat
```

或者，更简单、完全不涉及 C++ 工具链的做法是改用
[Ollama](https://ollama.com)：`COGNIVORE_LLM_PROVIDER=auto`（默认值）
会先尝试连接本地运行的 Ollama 服务器，然后才回退到 GGUF 路径或
`FakeLLMBackend`：

```bash
ollama pull qwen2.5:3b   # 任何经过指令微调的模型都可以用
cognivore chat           # 会自动识别正在运行的 Ollama 服务器
```

音频/视频工具需要单独的 extras：`pip install -e ".[audio,video]"`（或
使用 `.[all]` 安装全部内容，包括 GGUF）。所有配置项详见 `.env.example`。

视频工具中的屏幕文字提取功能（`analyze_video`）还需要
[Tesseract](https://github.com/tesseract-ocr/tesseract) OCR *二进制程
序* 本身——`video` extra 拉取的 `pytesseract` 包只是它的一层薄封装，
缺少这个二进制程序时 OCR 会静默地返回空文本（场景检测和时间戳不受影
响，仍能照常工作，因为那部分完全由 OpenCV 完成）：

```bash
sudo apt install tesseract-ocr        # Debian/Ubuntu
brew install tesseract                # macOS
# Windows: https://github.com/UB-Mannheim/tesseract/wiki
```

Docker 镜像已经内置了它，无需额外安装。

### Docker

该镜像采用多阶段构建（先编译原生 C++ 扩展，然后为精简的运行时镜像丢弃
编译器），以非 root 用户运行，并内置 `HEALTHCHECK` —— 借助 GitHub
Actions 的 Docker 层缓存构建一次，并在每次 push 时进行端到端验证（服务
器确实能响应 `/api/health`、容器报告为 `healthy`、以非 root 身份运
行），所以它不只是“能构建”，而是“经过验证”。

**最简单的方式 —— 无需 Python，无需安装 Ollama，在 Windows、macOS 和
Linux 上表现一致：**

```bash
docker compose up -d --build
docker compose exec ollama ollama pull qwen2.5:3b   # 仅需一次，约 2GB
```

然后打开 <http://127.0.0.1:8420>。`docker-compose.yml` 同样把 Ollama
运行在它*自己*的容器中，因此除了 Docker 本身之外，主机上什么都不用安
装；Cognivore 通过 compose 网络以服务名（`http://ollama:11434`）访问
它，这完全绕开了 Windows/macOS/Linux 之间常见的主机网络差异。即使不执
行那一次性的 `ollama pull`，Cognivore 依然能正常启动并运行 —— 只是在
有可用模型之前会回退到离线的 `FakeLLMBackend` 演示模式。

compose 方案还设置了
`COGNIVORE_SEED_DEMO_KB=true`，因此全新的知识
库会自动填充两份内置的演示公司手册（英文 + 俄
文——价格方案、SLA、安全、退款政策、支持常见问
题），而不是打开一个空的拖放区。它只会填充 *空
* 的知识库：一旦你导入了自己的文档，这就永久变
成空操作。可以在 `docker-compose.yml` 中将其设
为 `false`（或使用
`docker run -e COGNIVORE_SEED_DEMO_KB=false`）
以空知识库启动，也可以随时运行
`cognivore seed-demo`，把同样的演示文档添加到
已有的知识库中。

**已经在主机上运行了 Ollama，或者只想要一个容器？**

```bash
docker build -t cognivore .
docker run -d -p 8420:8420 -v cognivore-data:/data \
  -e COGNIVORE_OLLAMA_HOST=http://host.docker.internal:11434 \
  --add-host=host.docker.internal:host-gateway \
  cognivore
```

`host.docker.internal` 在 Docker Desktop（Windows/macOS）上会自动提
供；上面那条显式的 `--add-host` 正是让同一条命令在普通 Linux 上也能生
效的原因，否则这个名字在那里无法解析。

该镜像内置了音频/视频工具（`faster-whisper`、OpenCV，以及用于屏幕文
字 OCR 的 Tesseract），但*不*包含
`llama-cpp-python` —— 它是有意选择通过普通 HTTP 与 Ollama 通信来处理
LLM，而不是在进程内加载 GGUF 文件，因为 llama-cpp-python 并非对每个平
台都提供预构建的 wheel，而且需要运行时阶段并不具备的编译器。仍然想在容
器内进行进程内 GGUF 推理？可以在最终阶段中添加 `build-essential`，并把
其中的 `pip install` 换回 `[all]` extra。

**预构建镜像（完全无需构建步骤）：** 带标签的发布版本会以多架构形式
（amd64 + arm64 —— 包括 Apple Silicon 和 Raspberry Pi）发布到 GHCR，
构建流程见
[`.github/workflows/docker-publish.yml`](../../.github/workflows/docker-publish.yml)：

```bash
docker pull ghcr.io/anton-sergeev-ea/cognivore:latest
```

## 使用方法

**CLI**（完整列表见 `cognivore --help`）：

```bash
cognivore chat                              # 交互式 REPL
cognivore ingest ./docs                     # 递归索引 .txt/.md/.markdown/.rst 文件
cognivore seed-demo                         # 添加内置的英文+俄文演示公司手册
cognivore serve --host 0.0.0.0 --port 8420  # 网页界面 + REST/SSE API
cognivore bench                             # 索引构建/搜索基准测试（见下文）
```

**网页界面**（运行 `cognivore serve`，然后打开打印出来的 URL）：具备实
时 token 流式输出的聊天界面，可拖放上传 `.txt`/`.md` 文件以及音频/视频
文件的区域，语言切换器（9 种语言），三种主题（深色/浅色/aurora），以
及针对某个回答智能体所做每一次工具调用的追踪视图（点击某一步即可查看
完整、未截断的工具输出）。

**REST/SSE API**，在 `cognivore serve` 运行之后：

```bash
curl http://127.0.0.1:8420/api/health

curl -X POST http://127.0.0.1:8420/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is 12 * 7?"}'

curl -N "http://127.0.0.1:8420/api/chat/stream?message=Summarize+the+ingested+docs"

curl -X POST http://127.0.0.1:8420/api/ingest/file -F "file=@./notes.md"
```

**作为库使用**，而不通过 CLI 或 API（见 `examples/`）：

```python
from cognivore.bootstrap import build_agent
from cognivore.config import get_settings

agent = build_agent(get_settings())
result = agent.run("What's 15% of 5000?")
print(result.answer)
```

## 性能基准

以下数字来自在一台 2 核 CI 级机器上运行的
`python benchmarks/bench_index.py`（`Dockerfile` 生成的镜像是可移植
的；你实际使用的笔记本电脑几乎肯定拥有更多核心，这一点很重要 —— 详见
下文）。使用的是随机、均匀分布的 384 维向量，这对于近似搜索而言接近
*最差情况*（不存在真实的簇结构，因此“最近”邻居只是比随机点略微更近一
点）；在真实的嵌入向量上，相同 `ef` 下的 recall 会明显更高。

**构建时间**（3,000 个向量）—— 在这里原生扩展的优势毫无疑问：

| 索引 | 构建时间 | 相对 NumPy 回退方案 |
|---|---|---|
| `FlatIndexPy`（NumPy，循环中调用 `.add()`） | 0.615s | 1x |
| `FlatIndex`（C++） | 0.027s | **快约 23 倍** |

**Recall 与速度是一个可调节的旋钮，而非单一数值**（`NSWIndex`，
n=3,000）：

| `ef` | 毫秒/查询 | Recall@10 |
|---|---|---|
| 50  | 0.24 | 0.64 |
| 150（默认值） | 0.46 | 0.93 |
| 400 | 0.78 | 1.00 |

**搜索延迟与集合规模的关系** —— 精确的线性扫描（即使经过
AVX2+OpenMP 加速）在集合规模足够大、扫描本身成为瓶颈之前都是*没问题
的*；这个临界点正是近似图索引理论上应该开始占优的地方：

| n | `FlatIndex` 毫秒/查询 | `NSWIndex` 毫秒/查询（ef=150） | 加速比 |
|---|---|---|---|
| 1,000 | 0.036 | 0.240 | 0.2x（暴力搜索更胜一筹 —— 集合太小） |
| 10,000 | 0.340 | 0.798 | 0.4x |
| 50,000 | 1.561 | 1.291 | 1.2x |

请按照这张表实际反映的内容去理解，而不是按照能讲出更好故事的方式去理
解：在这些规模下、在 2 核上，SIMD + 并行的暴力扫描*与从零实现的近似索
引相当，甚至更快*。这是真实且有充分文献记录的 ANN 行为 —— 手写的单层
NSW 图在每一步上的开销（堆操作、跨图的随机内存访问、已访问集合的记录
维护）明显高于对缓存友好的线性扫描所享有的开销，而多层 HNSW
（Faiss/hnswlib 采用的方案，本项目尚未实现 —— 见路线图）正是为了在更
大规模上专门拉大这一差距而存在的。诚实的结论是：本项目的 ANN 索引正确
地展示了其*数据结构与算法*本身（见 `tests/test_index.py` 中的 recall
回归测试），而它真正开始“物有所值”的临界点会随着核心数量、`ef` 以及
你的真实嵌入向量的聚类程度而变化 —— 这不是一个“永远更快”的普适性断
言，本 README 也不会假装它是。

## 测试

```bash
pip install -e ".[dev]"
pytest --cov                 # 71 个测试：计算器安全性、分块（chunking），
                              # 原生索引与 Python 索引的一致性、NSW recall，
                              # 智能体循环、RAG 存储、FastAPI 端点
ruff check . && ruff format --check .
mypy -p cognivore
```

以上每一项检查正是 CI（`.github/workflows/ci.yml`）在
Ubuntu/macOS/Windows 与 Python 3.10-3.12 上运行的内容；在矩阵中没有
C++ 工具链的那一环上，仅依赖原生扩展的测试会自行跳过（而不是失败），
这与软件包本身的回退行为保持一致。

## 项目结构

见 [docs/architecture.md](../architecture.md)（英文）了解架构图，以及
两项最重大设计决策背后的理由（使用基于文本的 ReAct 循环而非特定厂商的
function calling，以及使用手写索引而非 Faiss/hnswlib）。

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

## 许可证

MIT 许可证 —— 详见 [LICENSE](../../LICENSE)。

## 联系方式

由 **Sergeev Anton** 开发（[GitHub](https://github.com/Anton-Sergeev-EA)，[avsergeev1981@gmail.com](mailto:avsergeev1981@gmail.com)）。
欢迎贡献 —— 详见 [CONTRIBUTING.md](../../CONTRIBUTING.md)。

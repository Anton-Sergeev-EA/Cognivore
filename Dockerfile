# Multi-stage build: compile the native extension in a full build image,
# then copy the installed package into a slim runtime image so the final
# image doesn't carry a C++ compiler around.

FROM python:3.11-slim AS builder
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /build
COPY pyproject.toml setup.py ./
COPY native ./native
COPY src ./src
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir build \
    && python -m build --wheel

FROM python:3.11-slim
LABEL org.opencontainers.image.title="cognivore" \
      org.opencontainers.image.description="Local-first multimodal agent framework" \
      org.opencontainers.image.source="https://github.com/Anton-Sergeev-EA/cognivore" \
      org.opencontainers.image.licenses="MIT"

WORKDIR /app

# tesseract-ocr is the native OCR *binary* that pytesseract (installed below
# via the `video` extra) shells out to -- the Python package alone can't do
# OCR without it. Without this, on-screen text extraction in `analyze_video`
# silently returns "" (the scene-detection/timestamps part still works, since
# that's pure OpenCV). eng+rus matches this project's two primary languages
# (see COGNIVORE_OCR_LANGUAGES / Settings.ocr_languages) -- confirmed live,
# requesting a language whose trained-data package isn't installed doesn't
# error out, it just quietly mis-recognizes that script as look-alike Latin
# letters (e.g. Cyrillic "Контейнеры" read as "KoHTewHepbi"), which reads
# like a low-quality scan rather than a missing language pack. So: if a
# deployment's on-screen text uses another language, add its
# `tesseract-ocr-<lang>` package here *and* that language to
# COGNIVORE_OCR_LANGUAGES, or it'll get silently garbled the same way.
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-rus \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /build/dist/*.whl /tmp/
# Deliberately `[audio,video]`, not `[all]`. `[all]` also pulls in `llm`
# (llama-cpp-python), which -- confirmed live, not hypothetical -- has no
# prebuilt wheel for every platform/Python combination and falls back to
# compiling from source via CMake/scikit-build-core. This runtime stage
# has no compiler (that's the whole point of the multi-stage build), so
# that install fails outright rather than just being slow.
#
# This isn't a real loss: the documented ways to run this image (compose,
# or a single container pointed at a host Ollama) both talk to Ollama
# over plain HTTP, which needs no native code in this image at all.
# llama-cpp-python is for loading a GGUF file *in-process* instead --
# a valid choice on bare metal, but a poor fit for "one image, any OS,
# any CPU," since it means compiling (or shipping a prebuilt wheel for)
# every target architecture. Add `build-essential` back to this stage
# and switch to `[all]` below if you specifically want that instead of
# Ollama.
# (Resolving the glob into a variable first, rather than writing
# `/tmp/*.whl[audio,video]` directly, avoids the shell parsing the
# brackets themselves as a glob character class appended to the
# wildcard.)
RUN WHEEL="$(ls /tmp/*.whl)" \
    && pip install --no-cache-dir "${WHEEL}[audio,video]" \
    && rm -f /tmp/*.whl

# HF_HOME redirects faster-whisper's Hugging Face model cache into the
# already-persisted /data volume (same reasoning as `ollama-data` in
# docker-compose.yml for the LLM: without this, the cache would live under
# the container's throwaway home directory, and the ~75MB-1GB Whisper
# model would be re-downloaded from Hugging Face on every rebuild/recreate
# instead of once.
ENV COGNIVORE_HOST=0.0.0.0 \
    COGNIVORE_PORT=8420 \
    COGNIVORE_DATA_DIR=/data \
    HF_HOME=/data/hf-cache

# Run as an unprivileged user rather than root -- this is a network-facing
# service (even if usually only reachable on localhost/a private compose
# network), so there's no reason to run it as root just because the base
# image defaults there. /data is created and owned by that user *before*
# VOLUME is declared so a fresh named volume inherits the right owner
# instead of showing up as root-owned and unwritable on first run.
RUN useradd --create-home --shell /usr/sbin/nologin --uid 1000 cognivore \
    && mkdir -p /data \
    && chown -R cognivore:cognivore /app /data
USER cognivore

VOLUME ["/data"]
EXPOSE 8420

# Lets `docker ps`, `docker compose ps`, and orchestrators (Swarm/Kubernetes
# with a translated probe) see "starting up" vs "actually serving requests"
# as two different states, rather than only "process is running" -- useful
# the moment there's more than one container to keep track of. Uses the
# stdlib rather than curl/wget so the final image doesn't need either
# installed just for this.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c 'import os, urllib.request as u; u.urlopen("http://127.0.0.1:%s/api/health" % os.environ.get("COGNIVORE_PORT", "8420"), timeout=3)'

ENTRYPOINT ["cognivore"]
CMD ["serve"]

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
COPY --from=builder /build/dist/*.whl /tmp/
# `[all]` pulls in the audio/video/llm extras; drop it for a smaller image
# if you only need the RAG + web chat surface without local media tools.
# (Resolving the glob into a variable first, rather than writing
# `/tmp/*.whl[all]` directly, avoids the shell parsing `[all]` itself as a
# glob character class appended to the wildcard.)
RUN WHEEL="$(ls /tmp/*.whl)" \
    && pip install --no-cache-dir "${WHEEL}[all]" \
    && rm -f /tmp/*.whl

ENV COGNIVORE_HOST=0.0.0.0 \
    COGNIVORE_PORT=8420 \
    COGNIVORE_DATA_DIR=/data

VOLUME ["/data"]
EXPOSE 8420

ENTRYPOINT ["cognivore"]
CMD ["serve"]

# Security Policy

Cognivore runs entirely on your own machine by default: no telemetry, and
the only outbound network calls are the ones you opt into (downloading a
model from Hugging Face for the semantic embedder, or a GGUF model you
supply yourself). Still, a few things are worth calling out explicitly:

- **The calculator tool never uses `eval`.** Expressions are parsed into a
  Python AST and only a small allow-listed set of node types is evaluated
  (see `src/cognivore/tools/calculator.py`). This matters specifically
  because tool arguments in an agent framework can originate from a
  document the LLM read, not just the end user, so tool implementations are
  held to the same bar as code accepting untrusted input anywhere else.
- **The web API has no authentication.** `cognivore serve` binds to
  `127.0.0.1` by default; if you change `COGNIVORE_HOST` to expose it on a
  network, put a reverse proxy with auth in front of it -- there is
  currently no login system.
- **Uploaded files** (`/api/ingest/file`, `/api/media/audio`,
  `/api/media/video`) are written to a temporary file, processed, and
  deleted; they are not persisted beyond the ingested text itself.

## Reporting a vulnerability

Please open a [GitHub Security Advisory](https://github.com/Anton-Sergeev-EA/cognivore/security/advisories/new)
or email avsergeev1981@gmail.com rather than filing a public issue.

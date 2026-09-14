## What this changes and why

<!-- The "why" specifically -- the diff already shows the "what." -->

## How it was verified

<!--
Before opening: `ruff check .`, `ruff format --check .`, `mypy -p cognivore`,
`pytest --cov` must all pass locally (CI runs the same four across
Linux/macOS/Windows, Python 3.10-3.12) -- see CONTRIBUTING.md.
-->

- [ ] Added or updated tests covering this change
- [ ] `pytest --cov` passes locally
- [ ] `ruff check .` / `ruff format --check .` / `mypy -p cognivore` pass locally
- [ ] Docs updated if this changes a documented command, endpoint, or config setting (`README.md` and, for anything user-facing, its 8 translations under `docs/i18n/`)

## Anything reviewers should look at closely

<!-- A tricky edge case, a design tradeoff you're not 100% sure about, a
place you deliberately didn't handle something -- optional, but this is
the single most useful section for someone reviewing agent/RAG/native-index
code they didn't write. Delete if there's genuinely nothing here. -->

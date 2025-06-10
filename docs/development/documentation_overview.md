# Documentation Overview

<!-- Consolidated from the old `documentation/README.md`. -->

This project's documentation stack combines several tools:

| Purpose                | Tool            | Invocation                                         |
| ---------------------- | --------------- | -------------------------------------------------- |
| General guides & refs  | Markdown        | `docs/**/*.md` rendered by **MkDocs**              |
| Notebook docs          | **NBDoc**       | `python -m nbdoc build notebooks/ -o docs/generated/notebooks` |
| API docs               | **pdoc**        | `python -m pdoc --html --output-dir docs/generated/api src` |
| Site build             | **MkDocs**      | `mkdocs serve` (local) / GH-Actions (CI)           |

Best-practice checklist:

1. Document as you code – PRs should update relevant pages.
2. Keep examples runnable; code blocks are doctested in CI via `mkdocs build --strict`.
3. Prefer small, modular pages over giant markdown files; cross-link liberally.
4. Auto-generated content under `docs/generated/` is **not** committed – build it locally or via CI.

# Kensaku CLI Architecture & Evolution Tasks

This roadmap outlines the tasks to evolve **Kensaku** from a basic semantic search tool into a production-grade, 100% local, in-process Q&A and hybrid retrieval CLI based on [qa_embeddings_onnx_architecture.md](file:///Users/darkoorovchanec/Documents/workplace/playground/oni/kensaku/qa_embeddings_onnx_architecture.md), tailored specifically for CLI performance and developer workflow.

---

## Task Inventory

| Task ID | Title | Priority | Target Components |
| :--- | :--- | :--- | :--- |
| [Task 01](file:///Users/darkoorovchanec/Documents/workplace/playground/oni/kensaku/tasks/task-01-sqlite-fts5-stemmer.md) | **SQLite FTS5 Porter Stemmer & Sync Triggers** | High | [`src/db/sqliteManager.js`](file:///Users/darkoorovchanec/Documents/workplace/playground/oni/kensaku/src/db/sqliteManager.js) |
| [Task 02](file:///Users/darkoorovchanec/Documents/workplace/playground/oni/kensaku/tasks/task-02-table-shielding-chunker.md) | **Table Shielding, Sentence Boundaries & Breadcrumbs** | High | [`src/pipeline/chunker.js`](file:///Users/darkoorovchanec/Documents/workplace/playground/oni/kensaku/src/pipeline/chunker.js) |
| [Task 03](file:///Users/darkoorovchanec/Documents/workplace/playground/oni/kensaku/tasks/task-03-transformers-v3-bge-embeddings.md) | **Upgrade to Transformers.js v3 & BGE Asymmetric Embeddings** | High | [`package.json`](file:///Users/darkoorovchanec/Documents/workplace/playground/oni/kensaku/package.json), [`src/pipeline/embeddings.js`](file:///Users/darkoorovchanec/Documents/workplace/playground/oni/kensaku/src/pipeline/embeddings.js) |
| [Task 04](file:///Users/darkoorovchanec/Documents/workplace/playground/oni/kensaku/tasks/task-04-cross-encoder-reranking.md) | **Stage-2 Batched Cross-Encoder Reranking** | High | [`src/services/searchEngine.js`](file:///Users/darkoorovchanec/Documents/workplace/playground/oni/kensaku/src/services/searchEngine.js) |
| [Task 05](file:///Users/darkoorovchanec/Documents/workplace/playground/oni/kensaku/tasks/task-05-local-slm-generation.md) | **In-Process Local ONNX SLM with Terminal Streaming** | Medium | [`src/services/llmClient.js`](file:///Users/darkoorovchanec/Documents/workplace/playground/oni/kensaku/src/services/llmClient.js), [`src/workers/llmWorker.js`](file:///Users/darkoorovchanec/Documents/workplace/playground/oni/kensaku/src/workers/llmWorker.js) |
| [Task 06](file:///Users/darkoorovchanec/Documents/workplace/playground/oni/kensaku/tasks/task-06-cli-ux-options.md) | **CLI Flags, First-Run Download UX & Diagnostics** | Medium | [`bin/kensaku.js`](file:///Users/darkoorovchanec/Documents/workplace/playground/oni/kensaku/bin/kensaku.js), [`src/index.js`](file:///Users/darkoorovchanec/Documents/workplace/playground/oni/kensaku/src/index.js) |
| [Task 07](file:///Users/darkoorovchanec/Documents/workplace/playground/oni/kensaku/tasks/task-07-remote-endpoints-config.md) | **Remote TEI/TGI Endpoints & CLI Config Command** | High | [`src/config.js`](file:///Users/darkoorovchanec/Documents/workplace/playground/oni/kensaku/src/config.js), [`bin/kensaku.js`](file:///Users/darkoorovchanec/Documents/workplace/playground/oni/kensaku/bin/kensaku.js) |

---

## Architectural Principles for CLI

1. **Preserve CLI Strengths**:
   - Keep multi-workspace support (`kensaku add/search/ask workspace <name>`).
   - Retain sub-millisecond mtime staleness checks ([`checkWorkspaceStaleness()`](file:///Users/darkoorovchanec/Documents/workplace/playground/oni/kensaku/src/index.js#L125-L172)).
   - Preserve line number references (`#L40-L55`) in chunk records for clear terminal citations.
   - Retain `.gitignore` filtering.
2. **Dual-Path Generation**:
   - Provide zero-config in-process ONNX generation (`onnx-community/Qwen2.5-1.5B-Instruct` or `0.5B`) for 100% offline, air-gapped environments.
   - Keep native external providers (Ollama, Anthropic, Gemini, OpenAI) for users wanting GPU acceleration via Ollama or cloud models.
3. **Streaming UX**:
   - Always stream tokens in real-time to stdout (`process.stdout.write`) rather than buffering responses.
4. **Hybrid Compute & Zero-Cold-Start Support**:
   - Run 100% in-process local ONNX by default.
   - Support persistent remote endpoints for Embeddings (TEI), Reranking (TEI), and Generation (TGI / Ollama) via `kensaku config set` or env vars to eliminate model-loading cold starts on demand.

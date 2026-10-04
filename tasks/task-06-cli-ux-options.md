# Task 06: CLI UX, Model Download Progress & Self-Diagnostics

## Objective
Enhance the CLI user experience in [`bin/kensaku.js`](file:///Users/darkoorovchanec/Documents/workplace/playground/oni/kensaku/bin/kensaku.js), providing clean terminal download progress bars during first-run ONNX model downloads, adding new command flags (`--no-rerank`, `--local`), and updating self-diagnostics in `kensaku test`.

---

## Background & Problem
1. **First-Run Hang Appearance**: When a user first executes a command requiring an ONNX model (`bge-small` 34 MB, `bge-reranker` 140 MB, or `Qwen2.5` 1.1 GB), the terminal may appear frozen if download progress is not displayed.
2. **Missing CLI Flags**: The CLI needs flags to toggle reranking (`--no-rerank`) and force local inference (`--local` or `--provider=local`).
3. **Diagnostics & Skill Synchronization**: `kensaku test` must verify the upgraded FTS5 stemmer, BGE embeddings, and local model readiness.

---

## Technical Specifications

### 1. Download Progress Callbacks
Transformers.js pipelines support `progress_callback`:
```javascript
const pipelineOptions = {
  dtype: 'q8',
  progress_callback: (progress) => {
    if (progress.status === 'progress') {
      const pct = Math.round((progress.loaded / progress.total) * 100);
      process.stdout.write(`\r[Kensaku] Downloading ${progress.file}: ${pct}%`);
    } else if (progress.status === 'done') {
      process.stdout.write(`\r[Kensaku] Downloaded ${progress.file} ✅\n`);
    }
  }
};
```
Wrap this in a reusable terminal reporter.

### 2. New CLI Flags in `bin/kensaku.js`
Support the following options:
* `kensaku search "<query>" [--no-rerank] [--limit=<n>]`
* `kensaku ask "<question>" [--local] [--provider=<p>] [--model=<m>]`
* `kensaku test` (runs diagnostics on: SQLite `sqlite-vec`, FTS5 Porter stemmer, BGE embedder, BGE reranker, and local/remote LLM health).

### 3. Update Documentation & Skills
Update [`SKILL.md`](file:///Users/darkoorovchanec/Documents/workplace/playground/oni/kensaku/SKILL.md) and [`bin/kensaku.js`](file:///Users/darkoorovchanec/Documents/workplace/playground/oni/kensaku/bin/kensaku.js) help output to highlight zero-configuration offline Q&A.

---

## Verification & Acceptance Criteria
- [ ] First-run model downloads show an interactive progress bar rather than a blank prompt.
- [ ] `kensaku test` runs and verifies all 5 subsystems:
  1. `sqlite-vec` vector extension
  2. FTS5 BM25 with English Porter stemmer
  3. BGE embedder (384-dim normalized)
  4. BGE reranker scoring
  5. Local/remote LLM readiness
- [ ] `SKILL.md` matches the final CLI capabilities.

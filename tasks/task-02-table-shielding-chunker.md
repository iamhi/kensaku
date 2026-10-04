# Task 02: Table Shielding, Sentence Boundaries & Contextual Breadcrumbs

## Objective
Upgrade the document chunking pipeline in [`src/pipeline/chunker.js`](file:///Users/darkoorovchanec/Documents/workplace/playground/oni/kensaku/src/pipeline/chunker.js) to isolate and shield Markdown tables, utilize sentence-boundary segmentation via `Intl.Segmenter`, and inject contextual breadcrumbs while preserving accurate line number tracking for CLI citations.

---

## Background & Problem
1. **Broken Tables**: The current sliding window chunker splits chunks when `maxChunkChars` (1200) is exceeded. A financial, feature comparison, or API parameter table is frequently sliced across arbitrary rows, destroying semantic coherence.
2. **Arbitrary Cuts**: Chunks are split on lines or character limits rather than sentence boundaries, occasionally cutting mid-thought.
3. **Missing Document Context in Chunks**: While `indexWorkspace` prepends the section path when calling the embedder, the stored chunk text in SQLite lacks the full breadcrumb context (`[Context: Document Title > Section]`), degrading BM25 keyword matching on topic names.

---

## Technical Specifications

### 1. Markdown Table Shielding
Isolate Markdown tables before splitting sections:
```javascript
const tableRegex = /(\|[^\n]+\|\n\|[-:\s|]+\|\n(?:\|[^\n]+\|\n?)+)/g;
```
- Tables are extracted as single, atomic chunk blocks (up to 800 tokens / 3000 chars).
- Do not split tables across chunk boundaries unless a single table exceeds maximum allowable limits.

### 2. Sentence-Bounded Chunking via `Intl.Segmenter`
For prose sections, use standard V8 `Intl.Segmenter`:
```javascript
const segmenter = new Intl.Segmenter('en', { granularity: 'sentence' });
const sentences = Array.from(segmenter.segment(proseText)).map((s) => s.segment);
```
- Group sentences up to ~350–450 tokens (~300 words).
- Retain a 1–2 sentence overlap between adjacent chunks.

### 3. Contextual Breadcrumbs & Line Tracking
- Prepend contextual breadcrumbs to the stored chunk:
  ```markdown
  [Context: Document Title > Section Name]
  <chunk content>
  ```
- **CLI Requirement**: Track exact file lines (`lineStart`, `lineEnd`) so `kensaku search` output continues to display clickable file links (e.g. `guide.md#L45-L68`).

---

## Verification & Acceptance Criteria
- [ ] Ingesting a document with a 40-row Markdown table keeps the table intact in a single chunk.
- [ ] Chunks break cleanly at sentence ends (`. `, `? `, `! `) rather than arbitrary character offsets.
- [ ] Stored chunks contain `[Context: ...]` headers, improving BM25 search when querying by section title.
- [ ] Line start and line end tracking remain accurate.
- [ ] Unit test in `test/chunker.test.js` verifies table isolation and sentence boundary preservation.

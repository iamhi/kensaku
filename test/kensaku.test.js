import { describe, it, before, after } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import { chunkMarkdown, hashContent } from '../src/pipeline/chunker.js';
import { embedText, embedBatch } from '../src/pipeline/embeddings.js';
import {
  addWorkspace,
  removeWorkspace,
  indexWorkspace,
  searchWorkspace,
  closeAllDatabases
} from '../src/index.js';

describe('Kensaku Semantic Search Engine', () => {
  let tempWorkspace;
  const testCategory = 'test_workspace_' + Date.now();

  before(async () => {
    tempWorkspace = await fs.mkdtemp(path.join(os.tmpdir(), 'kensaku-test-'));

    // Create sample markdown documents
    await fs.writeFile(
      path.join(tempWorkspace, 'architecture.md'),
      `# System Architecture

## Overview
The architecture is designed as a modular vertical slice system using Node.js and SQLite.
It supports fast in-memory query processing and vector similarity searching.

## Storage Layer
Storage is powered by SQLite with sqlite-vec extension for 384-dimensional dense vectors.
FTS5 full-text indexing enables authentic BM25 lexical ranking.
`,
      'utf8'
    );

    await fs.writeFile(
      path.join(tempWorkspace, 'ai_research.md'),
      `# Frontier AI Research 2026

## Transformer Models
Recent breakthroughs in quantized small language models enable high-performance edge inference.
ONNX runtime accelerates local execution without GPU requirements.

## Mathematical Formulations
Attention mechanisms scale with sequence length according to the formula:
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
`,
      'utf8'
    );
  });

  after(async () => {
    closeAllDatabases();
    removeWorkspace(testCategory);
    try {
      await fs.rm(tempWorkspace, { recursive: true, force: true });
    } catch {}
  });

  it('correctly chunks markdown and tracks line numbers and hashes', () => {
    const md = `# Document Title\n\n## Section 1\nLine 1\nLine 2\n\n## Section 2\nLine 5`;
    const chunks = chunkMarkdown(md, 'test.md');
    assert.strictEqual(chunks.length, 3);
    assert.strictEqual(chunks[0].title, 'Document Title');
    assert.strictEqual(chunks[1].section, 'Section 1');
    assert.strictEqual(chunks[1].lineStart, 3);
    assert.strictEqual(chunks[2].section, 'Section 2');

    const h1 = hashContent(md);
    const h2 = hashContent(md);
    assert.strictEqual(h1, h2);
    assert.strictEqual(typeof h1, 'string');
  });

  it('generates 384-dimensional embeddings via @xenova/transformers', async () => {
    const vec = await embedText('Hello semantic search');
    assert.ok(vec instanceof Float32Array);
    assert.strictEqual(vec.length, 384);

    const batch = await embedBatch(['Text one', 'Text two']);
    assert.strictEqual(batch.length, 2);
    assert.strictEqual(batch[0].length, 384);
  });

  it('indexes a workspace category and executes hybrid search with RRF', async () => {
    addWorkspace(testCategory, tempWorkspace);

    const indexRes = await indexWorkspace(testCategory, { rootDir: tempWorkspace });
    assert.strictEqual(indexRes.scanned, 2);
    assert.ok(indexRes.updated >= 2);
    assert.ok(indexRes.totalChunks >= 4);

    // Incremental index test (nothing changed)
    const incRes = await indexWorkspace(testCategory, { rootDir: tempWorkspace });
    assert.strictEqual(incRes.scanned, 2);
    assert.strictEqual(incRes.updated, 0);
    assert.strictEqual(incRes.unchanged, 2);

    // Semantic search query
    const searchRes = await searchWorkspace(testCategory, 'vector database and sqlite storage', { limit: 3 });
    assert.ok(searchRes.results.length > 0);
    assert.strictEqual(searchRes.results[0].docPath, 'architecture.md');
    assert.ok(searchRes.results[0].score > 0);
    assert.ok(searchRes.results[0].lineStart > 0);

    // AI research query
    const aiRes = await searchWorkspace(testCategory, 'quantized transformer edge inference', { limit: 3 });
    assert.ok(aiRes.results.length > 0);
    assert.strictEqual(aiRes.results[0].docPath, 'ai_research.md');
  });
});

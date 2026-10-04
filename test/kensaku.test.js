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
  checkWorkspaceStaleness,
  getCompositeWorkspaceHash,
  closeAllDatabases
} from '../src/index.js';
import { reorderLostInTheMiddle, formatContextEnvelopes } from '../src/services/ragEngine.js';

describe('Kensaku Semantic Search & RAG Engine', () => {
  let tempWorkspace1;
  let tempWorkspace2;
  const testCategory1 = 'test_cat1_' + Date.now();
  const testCategory2 = 'test_cat2_' + Date.now();

  before(async () => {
    tempWorkspace1 = await fs.mkdtemp(path.join(os.tmpdir(), 'kensaku-test1-'));
    tempWorkspace2 = await fs.mkdtemp(path.join(os.tmpdir(), 'kensaku-test2-'));

    // Create sample markdown documents in workspace 1 (Architecture & DB)
    await fs.writeFile(
      path.join(tempWorkspace1, 'architecture.md'),
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

    // Create sample markdown documents in workspace 2 (Cooking & Recipes)
    await fs.writeFile(
      path.join(tempWorkspace2, 'carbonara.md'),
      `# Traditional Italian Cuisine

## Spaghetti alla Carbonara
To prepare classic Roman Carbonara, you need guanciale, fresh pecorino romano cheese, eggs, and black pepper.
Never use cream or garlic in an authentic Roman recipe.
Slowly render the guanciale until crispy before tossing with the al dente pasta.
`,
      'utf8'
    );
  });

  after(async () => {
    closeAllDatabases();
    removeWorkspace(testCategory1);
    removeWorkspace(testCategory2);
    try {
      await fs.rm(tempWorkspace1, { recursive: true, force: true });
      await fs.rm(tempWorkspace2, { recursive: true, force: true });
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

  it('indexes multiple workspaces and verifies composite hashing and staleness detection', async () => {
    addWorkspace(testCategory1, tempWorkspace1);
    addWorkspace(testCategory2, tempWorkspace2);

    const indexRes1 = await indexWorkspace(testCategory1, { rootDir: tempWorkspace1 });
    assert.strictEqual(indexRes1.scanned, 1);
    assert.ok(indexRes1.updated >= 1);
    assert.ok(indexRes1.compositeHash.length === 64);

    const indexRes2 = await indexWorkspace(testCategory2, { rootDir: tempWorkspace2 });
    assert.strictEqual(indexRes2.scanned, 1);
    assert.ok(indexRes2.compositeHash.length === 64);

    // Initial staleness check: should NOT be stale
    const staleCheck1 = await checkWorkspaceStaleness(testCategory1);
    assert.strictEqual(staleCheck1.isStale, false);
    assert.strictEqual(staleCheck1.compositeHash, indexRes1.compositeHash);

    // Modify a file: should become stale
    await fs.appendFile(path.join(tempWorkspace1, 'architecture.md'), '\n## Cache Layer\nRedis cache in-memory.\n');
    const staleCheckAfterMod = await checkWorkspaceStaleness(testCategory1);
    assert.strictEqual(staleCheckAfterMod.isStale, true);
    assert.strictEqual(staleCheckAfterMod.modifiedFiles.length, 1);

    // Re-index should clear staleness and update composite hash
    const reindexRes = await indexWorkspace(testCategory1, { rootDir: tempWorkspace1 });
    assert.strictEqual(reindexRes.updated, 1);
    assert.notStrictEqual(reindexRes.compositeHash, indexRes1.compositeHash);

    const staleCheckAfterReindex = await checkWorkspaceStaleness(testCategory1);
    assert.strictEqual(staleCheckAfterReindex.isStale, false);
  });

  it('searches across ALL registered workspaces simultaneously with global ranking', async () => {
    // Search across all workspaces
    const globalRes = await searchWorkspace('all', 'authentic roman guanciale recipe', { limit: 5 });
    assert.ok(globalRes.results.length > 0);
    assert.strictEqual(globalRes.results[0].workspace, testCategory2);
    assert.strictEqual(globalRes.results[0].docPath, 'carbonara.md');

    // Search for tech architecture across all workspaces
    const techRes = await searchWorkspace('all', 'sqlite and dense vector extension', { limit: 5 });
    assert.ok(techRes.results.length > 0);
    assert.strictEqual(techRes.results[0].workspace, testCategory1);
    assert.strictEqual(techRes.results[0].docPath, 'architecture.md');
  });

  it('implements Lost-in-the-Middle context reordering and XML envelope formatting', () => {
    const mockChunks = [
      { id: 1, docPath: 'a.md', lineStart: 1, lineEnd: 10, content: 'Chunk A (Best)', score: 0.95 },
      { id: 2, docPath: 'b.md', lineStart: 1, lineEnd: 10, content: 'Chunk B (Second)', score: 0.85 },
      { id: 3, docPath: 'c.md', lineStart: 1, lineEnd: 10, content: 'Chunk C (Third)', score: 0.75 },
      { id: 4, docPath: 'd.md', lineStart: 1, lineEnd: 10, content: 'Chunk D (Fourth)', score: 0.65 }
    ];

    const reordered = reorderLostInTheMiddle(mockChunks);
    assert.strictEqual(reordered.length, 4);
    // Highest score at beginning
    assert.strictEqual(reordered[0].id, 1);
    // Second highest at the end (Lost-in-the-Middle mitigation)
    assert.strictEqual(reordered[3].id, 2);

    const envelopes = formatContextEnvelopes(reordered);
    assert.ok(envelopes.includes('<context>'));
    assert.ok(envelopes.includes('<document id="1"'));
    assert.ok(envelopes.includes('Chunk A (Best)'));
    assert.ok(envelopes.includes('</context>'));
  });
});

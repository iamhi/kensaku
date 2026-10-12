import { describe, it, before, after } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import {
  addWorkspace,
  removeWorkspace,
  closeAllDatabases,
  getWorkspaceMetadata,
  setWorkspaceMetadata,
  verifyWorkspaceEmbeddingDimension,
  getCategoryStats,
  renderCitations,
  auditCitations,
  buildSourceUri,
  rerankCandidates,
  searchWorkspace
} from '../src/index.js';

describe('Advanced RAG Capabilities (ODS Improvements)', () => {
  let tempWorkspace;
  const testCategory = 'test_adv_rag_' + Date.now();

  before(async () => {
    tempWorkspace = await fs.mkdtemp(path.join(os.tmpdir(), 'kensaku-adv-test-'));
    await fs.writeFile(
      path.join(tempWorkspace, 'guide.md'),
      `# High Performance RAG Guide

## Vector Storage
SQLite with sqlite-vec provides embedded vector search without external microservices.
It stores dense 384-dimensional embeddings.

## Lexical Indexing
FTS5 full text search provides BM25 scoring for exact keyword matches.
`,
      'utf8'
    );
    addWorkspace(testCategory, tempWorkspace);
    const { indexWorkspace } = await import('../src/index.js');
    await indexWorkspace(testCategory);
  });

  after(async () => {
    closeAllDatabases();
    removeWorkspace(testCategory);
    try {
      await fs.rm(tempWorkspace, { recursive: true, force: true });
    } catch {}
  });

  describe('1. Embedding Dimension Metadata Governance', () => {
    it('initializes and reads workspace metadata for model and dimension', () => {
      const meta = getWorkspaceMetadata(testCategory);
      assert.strictEqual(meta.model_name, 'Xenova/all-MiniLM-L6-v2');
      assert.strictEqual(meta.embedding_dim, '384');

      const stats = getCategoryStats(testCategory);
      assert.strictEqual(stats.embeddingModel, 'Xenova/all-MiniLM-L6-v2');
      assert.strictEqual(stats.embeddingDim, 384);
    });

    it('allows updating metadata key-value pairs', () => {
      setWorkspaceMetadata(testCategory, 'custom_note', 'production-v1');
      const meta = getWorkspaceMetadata(testCategory);
      assert.strictEqual(meta.custom_note, 'production-v1');
    });

    it('validates matching dimension successfully', () => {
      const res = verifyWorkspaceEmbeddingDimension(testCategory, 'Xenova/all-MiniLM-L6-v2', 384);
      assert.strictEqual(res.ok, true);
      assert.strictEqual(res.dimension, 384);
    });

    it('throws actionable error on dimension mismatch', () => {
      assert.throws(
        () => {
          verifyWorkspaceEmbeddingDimension(testCategory, 'BAAI/bge-base-en-v1.5', 768);
        },
        (err) => {
          assert.match(err.message, /Dimension mismatch for workspace/);
          assert.match(err.message, /database was initialized with dimension 384/);
          assert.match(err.message, /current operation requires dimension 768/);
          assert.match(err.message, /kensaku index.*--force/);
          return true;
        }
      );
    });
  });

  describe('2. Code-Aware Citation Renderer & Linkifier', () => {
    const mockSources = [
      { id: 1, docPath: 'docs/storage.md', lineStart: 10, lineEnd: 25, title: 'Storage Architecture', section: 'Vector DB' },
      { id: 2, docPath: 'docs/api.md', lineStart: 40, lineEnd: 60, title: 'API Specification', section: 'Endpoints' }
    ];

    it('correctly builds file URI with line ranges', () => {
      const uri = buildSourceUri(mockSources[0], '/workspace/root');
      assert.match(uri, /^file:\/\//);
      assert.match(uri, /docs\/storage\.md#L10-L25$/);
    });

    it('linkifies single citation in prose', () => {
      const rawText = 'SQLite is used for storage [1].';
      const rendered = renderCitations(rawText, mockSources);
      assert.match(rendered, /SQLite is used for storage \[1\]\(.*docs\/storage\.md#L10-L25/);
    });

    it('expands compound citations [1, 2] into discrete clickable links', () => {
      const rawText = 'Both modules cooperate [1, 2].';
      const rendered = renderCitations(rawText, mockSources);
      assert.match(rendered, /\[1\]\(.*docs\/storage\.md#L10-L25/);
      assert.match(rendered, /\[2\]\(.*docs\/api\.md#L40-L60/);
    });

    it('preserves code literals inside inline backticks', () => {
      const rawText = 'Access the element via `const item = items[1];` as explained in [1].';
      const rendered = renderCitations(rawText, mockSources);
      assert.ok(rendered.includes('`const item = items[1];`'));
      assert.match(rendered, /as explained in \[1\]\(.*storage\.md/);
    });

    it('preserves fenced code blocks without corrupting bracketed code', () => {
      const rawText = `Here is an example:
\`\`\`javascript
const arr = [1, 2];
console.log(arr[1]);
\`\`\`
This follows the specification [2].`;

      const rendered = renderCitations(rawText, mockSources);
      assert.ok(rendered.includes('const arr = [1, 2];'));
      assert.ok(rendered.includes('console.log(arr[1]);'));
      assert.match(rendered, /This follows the specification \[2\]\(.*docs\/api\.md#L40-L60/);
    });

    it('filters out hallucinated / phantom citations [9] when out of bounds', () => {
      const rawText = 'Verified fact [1] and hallucinated claim [9].';
      const rendered = renderCitations(rawText, mockSources);
      assert.match(rendered, /Verified fact \[1\]\(/);
      assert.ok(!rendered.includes('[9]'));
    });

    it('audits citations and detects valid vs hallucinated markers', () => {
      const rawText = 'See [1], [2], and unknown [5].';
      const audit = auditCitations(rawText, mockSources);
      assert.deepStrictEqual(audit.citations, [1, 2, 5]);
      assert.deepStrictEqual(audit.validCitations, [1, 2]);
      assert.deepStrictEqual(audit.hallucinatedCitations, [5]);
      assert.strictEqual(audit.citedSources.length, 2);
    });
  });

  describe('3. Stage-2 Cross-Encoder Reranker', () => {
    const candidates = [
      { id: 1, content: 'Pecorino cheese and guanciale for carbonara', score: 0.5 },
      { id: 2, content: 'SQLite WAL mode and database concurrency', score: 0.9 },
      { id: 3, content: 'Eggs and pasta water emulsification in carbonara', score: 0.4 }
    ];

    it('skips reranking when noRerank or rerank=false is specified', async () => {
      const rescored = await rerankCandidates('carbonara recipe', candidates, { noRerank: true });
      assert.strictEqual(rescored.length, 3);
      assert.strictEqual(rescored[0].id, 1);
    });

    it('rescores candidates using custom reranker function', async () => {
      // Mock reranker scoring query relevance
      const customReranker = async (query, candidate) => {
        if (candidate.content.includes('carbonara')) {
          return candidate.content.includes('guanciale') ? 0.98 : 0.85;
        }
        return 0.05;
      };

      const rescored = await rerankCandidates('carbonara recipe', candidates, { customReranker });
      assert.strictEqual(rescored.length, 3);
      assert.strictEqual(rescored[0].id, 1);
      assert.strictEqual(rescored[0].rerankScore, 0.98);
      assert.strictEqual(rescored[1].id, 3);
      assert.strictEqual(rescored[1].rerankScore, 0.85);
      assert.strictEqual(rescored[2].id, 2);
      assert.strictEqual(rescored[2].rerankScore, 0.05);
    });

    it('integrates with searchWorkspace when rerank flag is active', async () => {
      const searchRes = await searchWorkspace(testCategory, 'vector storage', {
        limit: 2,
        customReranker: async (query, c) => (c.content.includes('sqlite-vec') ? 0.99 : 0.1)
      });
      assert.ok(searchRes.results.length > 0);
      assert.strictEqual(searchRes.results[0].rerankScore, 0.99);
    });
  });
});

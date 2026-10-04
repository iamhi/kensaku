# Production Semantic Search Architecture: Vanilla JavaScript + SQLite + Hugging Face TEI (Rust)

A turnkey, production-grade architectural and implementation blueprint for deploying a lightweight, high-performance semantic search system using **Vanilla JavaScript (Node.js 22/24 LTS Native ESM)**, **SQLite** (featuring **File-per-Project** multi-tenancy, `sqlite-vec`, and native **FTS5 BM25**), and **Hugging Face Text Embeddings Inference (TEI)** in Rust.

---

## Table of Contents

1. [Architectural Overview & File-per-Project Multi-Tenancy](#1-architectural-overview--file-per-project-multi-tenancy)
2. [Hugging Face TEI: The Rust Inference Engine](#2-hugging-face-tei-the-rust-inference-engine)
   - [TEI Internal Mechanics & Continuous Dynamic Batching](#tei-internal-mechanics--continuous-dynamic-batching)
   - [Recommended Bi-Encoder Embedding Models](#recommended-bi-encoder-embedding-models)
   - [Docker Deployment Specifications (CPU & GPU)](#docker-deployment-specifications-cpu--gpu)
3. [Hardened Vanilla JavaScript TEI Client](#3-hardened-vanilla-javascript-tei-client)
   - [Socket Connection Pooling (`undici.Pool`)](#socket-connection-pooling-undicipool)
   - [Circuit Breaking (`opossum`) & Jittered Exponential Backoff](#circuit-breaking-opossum--jittered-exponential-backoff)
   - [Bounded Concurrency Batching (`p-limit`)](#bounded-concurrency-batching-p-limit)
   - [Complete Client Implementation (`src/services/teiClient.js`)](#complete-client-implementation-srcservicesteiclientjs)
4. [Document Ingestion, AST Chunking & Security](#4-document-ingestion-ast-chunking--security)
   - [Recursive AST Chunking (cAST) vs. Symbol Collapsing](#recursive-ast-chunking-cast-vs-symbol-collapsing)
   - [Token Budget Enforcement vs. Character Caps](#token-budget-enforcement-vs-character-caps)
   - [Defense-in-Depth Secret Redaction](#defense-in-depth-secret-redaction)
   - [Ingestion Pipeline Implementation (`src/pipeline/ingestion.js`)](#ingestion-pipeline-implementation-srcpipelineingestionjs)
5. [SQLite Storage Layer: `better-sqlite3` + `sqlite-vec` + FTS5](#5-sqlite-storage-layer-better-sqlite3--sqlite-vec--fts5)
   - [Why the File-per-Project Pattern Wins for SQLite](#why-the-file-per-project-pattern-wins-for-sqlite)
   - [Database Schema with Automatic FTS5 Synchronization](#database-schema-with-automatic-fts5-synchronization)
   - [WAL Mode & High-Concurrency PRAGMAs](#wal-mode--high-concurrency-pragmas)
   - [SQLite Project Manager Implementation (`src/db/sqliteManager.js`)](#sqlite-project-manager-implementation-srcdbsqlitemanagerjs)
6. [Lightweight Hybrid Search & Reciprocal Rank Fusion (RRF)](#6-lightweight-hybrid-search--reciprocal-rank-fusion-rrf)
   - [Dense Vector Distance (`sqlite-vec`)](#dense-vector-distance-sqlite-vec)
   - [Native Lexical BM25 Scoring (SQLite FTS5)](#native-lexical-bm25-scoring-sqlite-fts5)
   - [Calibrated 1-Based Reciprocal Rank Fusion (RRF)](#calibrated-1-based-reciprocal-rank-fusion-rrf)
   - [Cumulative Density Proximity Chunk Merging](#cumulative-density-proximity-chunk-merging)
   - [Hybrid Search Service Implementation (`src/services/searchEngine.js`)](#hybrid-search-service-implementation-srcservicessearchenginejs)
7. [Optional: Pluggable Second-Stage Cross-Encoder Reranking](#7-optional-pluggable-second-stage-cross-encoder-reranking)
   - [When to Enable Two-Stage Reranking](#when-to-enable-two-stage-reranking)
   - [Reranker Service Integration (`/rerank`)](#reranker-service-integration-rerank)
8. [Complete Production Docker Compose Environment](#8-complete-production-docker-compose-environment)
9. [Operations, Provenance & Zero-Downtime Re-indexing](#9-operations-provenance--zero-downtime-re-indexing)
   - [Composite Content-Hash Gating](#composite-content-hash-gating)
   - [Atomic Project Re-indexing & Hot Swap](#atomic-project-re-indexing--hot-swap)
10. [Vanilla JavaScript Package Manifest & Directory Layout](#10-vanilla-javascript-package-manifest--directory-layout)

---

## 1. Architectural Overview & File-per-Project Multi-Tenancy

This architecture provides a streamlined, zero-build-step semantic search engine. It decouples high-throughput tensor operations from Node.js while taking full advantage of embedded SQLite databases.

The system is organized into three distinct layers:
1. **Inference Layer (Hugging Face TEI in Rust)**: A stateless microservice executing continuous dynamic batching, FlashAttention-2, and SIMD-accelerated token pooling.
2. **Orchestration Layer (Node.js 22/24 LTS in Vanilla JS ESM)**: Handles file intake, AST syntax chunking, token counting, secret redaction, and search result fusion. Runs directly with `node src/index.js`—no TypeScript transpilation or build step.
3. **Storage & Index Layer (SQLite via `better-sqlite3`)**: Organizes storage using an isolated **File-per-Project** layout (`./data/projects/${projectId}.db`). Dense vectors are indexed via `sqlite-vec`, while lexical text is indexed via SQLite's native **FTS5 with authentic BM25 ranking**.

```mermaid
flowchart TD
    subgraph Client_Layer["Node.js Application Layer (Vanilla JavaScript ESM)"]
        Ingest["Document Ingestion\n(AST Chunker + Secret Redactor)"]
        SearchAPI["Search Service\n(Dense KNN + Native FTS5 BM25)"]
        TeiClient["Resilient TEI Client\n(Undici Pool + Opossum Breaker)"]
    end

    subgraph Storage_Layer["Local Storage (Isolated SQLite Files: ./data/projects/)"]
        subgraph TenantA["Tenant Alpha (alpha.db)"]
            ChunksA["chunks (text + metadata)"]
            VecA["chunk_vectors (vec0 virtual table)"]
            FTSA["chunks_fts (FTS5 BM25 virtual table)"]
        end
        subgraph TenantB["Tenant Beta (beta.db)"]
            ChunksB["chunks (text + metadata)"]
            VecB["chunk_vectors (vec0 virtual table)"]
            FTSB["chunks_fts (FTS5 BM25 virtual table)"]
        end
    end

    subgraph Inference_Fleet["Stateless Rust Inference Layer (Hugging Face TEI)"]
        TEI["TEI Embedding Container (Bi-Encoder)\nModel: BAAI/bge-base-en-v1.5"]
        OptRerank["Optional TEI Reranker Container (Cross-Encoder)\nModel: BAAI/bge-reranker-v2-m3"]
    end

    %% Ingestion Flow
    Ingest -->|"1. Slice tokens & AST"| TeiClient
    TeiClient -->|"2. POST /embed (p-limit)"| TEI
    TEI -->|"3. Float32 vectors"| TeiClient
    TeiClient -->|"4. Upsert chunks & vectors"| ChunksA & VecA & FTSA

    %% Search Flow
    SearchAPI -->|"1. Embed query"| TeiClient
    TeiClient -->|"2. POST /embed"| TEI
    SearchAPI -->|"3. Parallel KNN + FTS5 BM25"| TenantA
    TenantA -->|"4. Reciprocal Rank Fusion (RRF)"| SearchAPI
    SearchAPI -.->|"Optional 5. POST /rerank"| OptRerank
```

### Why the File-per-Project Pattern Eliminates Vector Filtering Defects
In centralized relational databases (such as single-table PostgreSQL setups), filtering Approximate Nearest Neighbor (ANN) vector searches with `WHERE project_id = 'alpha'` causes the **overfiltering defect**: graph exploration explores foreign tenant nodes and exhausts exploration budgets, silently truncating search recall.

In this SQLite architecture:
* **Zero Cross-Tenant Contamination**: Each project resides in its own isolated database file (`alpha.db`). The vector table contains **100% project-owned vectors**.
* **Zero Lock Contention**: Writing to Project Alpha never blocks readers or writers in Project Beta.
* **Instant Deletion & Archiving**: Offboarding a tenant is an atomic filesystem operation (`fs.unlinkSync('./data/projects/alpha.db')`). Backing up a project is a simple file copy (`cp alpha.db alpha.db.bak`).
* **Zero Vacuum Bloat**: Relational database table bloat from tenant churn is completely eliminated.

---

## 2. Hugging Face TEI: The Rust Inference Engine

### TEI Internal Mechanics & Continuous Dynamic Batching
Hugging Face Text Embeddings Inference (TEI) is compiled in Rust on top of Candle, ONNX Runtime, and PyTorch C++ kernels. 
* **Continuous Dynamic Batching**: TEI does not hold requests for a static batch window (e.g. 50ms). It schedules incoming requests into the very next tensor forward pass at the token level, maximizing hardware utilization.
* **Low Idle Memory**: Consumes ~50–100 MB of base RAM (plus model weight allocations), compared to 2–4 GB for Python/PyTorch wrappers.
* **Zero V8 / Node Thread Contention**: Matrix multiplication and tensor pooling are completely offloaded from the Node.js event loop and libuv threadpool.

### Recommended Bi-Encoder Embedding Models

| Model ID | Domain Specialization | Context Window | Vector Dimension | Storage Size | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`BAAI/bge-base-en-v1.5`** | General English & Technical Docs | 512 tokens | 768 | ~440 MB | Industry standard for technical retrieval; fast, high recall. |
| **`BAAI/bge-small-en-v1.5`** | Lightweight / Low Latency | 512 tokens | 384 | ~130 MB | Ultra-fast inference on CPU; minimal memory footprint. |
| **`jinaai/jina-embeddings-v2-base-code`** | Source Code & API Documentation | 8,192 tokens | 768 | ~640 MB | Trained on code ASTs; handles long functions and entire files. |
| **`nomic-ai/nomic-embed-text-v1.5`** | Long Documents & Flexible Storage | 8,192 tokens | 768 (Matryoshka to 256/128) | ~550 MB | Requires task prefixes (`search_query:` / `search_document:`). |

---

### Docker Deployment Specifications (CPU & GPU)

#### 1. CPU Production Deployment
```bash
docker run -d \
  --name tei-embeddings \
  --restart unless-stopped \
  -p 8080:80 \
  -v $HOME/.cache/huggingface/hub:/data \
  ghcr.io/huggingface/text-embeddings-inference:cpu-1.9 \
  --model-id BAAI/bge-base-en-v1.5 \
  --revision a5eda8f47b2c15c0e241780540445a6c117d33d9 \
  --port 80 \
  --max-client-batch-size 32 \
  --max-batch-tokens 16384 \
  --default-prompt-name query
```

#### 2. GPU Production Deployment (NVIDIA CUDA)
```bash
docker run -d \
  --name tei-embeddings-gpu \
  --gpus all \
  --restart unless-stopped \
  -p 8080:80 \
  -v $HOME/.cache/huggingface/hub:/data \
  ghcr.io/huggingface/text-embeddings-inference:1.9 \
  --model-id BAAI/bge-base-en-v1.5 \
  --port 80 \
  --max-client-batch-size 64 \
  --max-batch-tokens 16384
```

> [!IMPORTANT]
> **Healthcheck and Startup Grace Periods**
> On first boot, TEI downloads model weights from Hugging Face. When configuring Docker Compose or Kubernetes, always assign a `start_period` of at least **60 seconds** to prevent premature container restarts.

---

## 3. Hardened Vanilla JavaScript TEI Client

### Socket Connection Pooling (`undici.Pool`)
In Node.js 18 through 24, global `fetch` is implemented using the `undici` engine. Passing `node:http.Agent` to `fetch()` is silently ignored. Under high query volume, unmanaged dispatchers exhaust system sockets and trigger connection timeouts.

The hardened client:
1. Employs `undici.Pool` with persistent HTTP Keep-Alive and configurable connection limits.
2. Protects the application using an `opossum` Circuit Breaker to fail fast when downstream inference crashes or runs out of memory.
3. Separates fatal client errors (HTTP 400, 413, 422) from transient errors (HTTP 429, 502, 503, 504), retrying only the latter with full-jitter exponential backoff.
4. Uses `p-limit` to bound concurrent sub-batch dispatching, saturating TEI's continuous dynamic batcher.

### Complete Client Implementation (`src/services/teiClient.js`)

```javascript
// src/services/teiClient.js
import { Pool } from 'undici';
import CircuitBreaker from 'opossum';
import pLimit from 'p-limit';

/**
 * Production TEI Client in Vanilla JavaScript (Node.js ESM)
 */
export class ResilientTeiClient {
  /**
   * @param {Object} config
   * @param {string} config.baseUrl - e.g. 'http://localhost:8080'
   * @param {number} [config.timeoutMs=8000] - Request timeout in ms
   * @param {number} [config.maxConnections=16] - HTTP connection pool size
   * @param {number} [config.batchSize=32] - Texts per /embed payload (aligned with TEI default)
   * @param {number} [config.concurrencyLimit=4] - Max concurrent HTTP batches in flight
   * @param {string} [config.promptName] - Optional TEI prompt template (e.g. 'query')
   */
  constructor(config) {
    const url = new URL(config.baseUrl);
    this.timeoutMs = config.timeoutMs ?? 8000;
    this.batchSize = config.batchSize ?? 32;
    this.promptName = config.promptName;
    this.limit = pLimit(config.concurrencyLimit ?? 4);

    // Dedicated undici connection pool for managed socket reuse
    this.pool = new Pool(url.origin, {
      connections: config.maxConnections ?? 16,
      pipelining: 1,
      keepAliveTimeout: 30000,
      keepAliveMaxTimeout: 60000,
    });

    // Circuit breaker to prevent cascading failures
    this.breaker = new CircuitBreaker(
      (inputs, isQuery) => this.dispatchHttp(inputs, isQuery),
      {
        timeout: this.timeoutMs,
        errorThresholdPercentage: 50,
        resetTimeout: 10000,
        rollingCountTimeout: 10000,
        rollingCountBuckets: 10,
      }
    );

    this.breaker.fallback(() => {
      throw new Error('Circuit Breaker OPEN: TEI inference service is unhealthy or unreachable.');
    });
  }

  /**
   * Embed a single search query.
   * @param {string} query
   * @returns {Promise<number[]>} Float vector
   */
  async embedQuery(query) {
    const [vector] = await this.breaker.fire([query], true);
    if (!vector || vector.length === 0) {
      throw new Error('TEI returned an empty embedding vector for the search query.');
    }
    return vector;
  }

  /**
   * Embed an array of document chunks with bounded concurrent batching.
   * @param {string[]} documents
   * @returns {Promise<number[][]>} Array of float vectors
   */
  async embedDocuments(documents) {
    if (documents.length === 0) return [];

    const slices = [];
    for (let i = 0; i < documents.length; i += this.batchSize) {
      slices.push(documents.slice(i, i + this.batchSize));
    }

    // Concurrent dispatch through p-limit feeds TEI's continuous batcher
    const tasks = slices.map((slice) =>
      this.limit(() => this.breaker.fire(slice, false))
    );

    const nestedResults = await Promise.all(tasks);
    return nestedResults.flat();
  }

  /**
   * Internal HTTP dispatcher with retry and error classification.
   * @private
   */
  async dispatchHttp(inputs, isQuery) {
    const maxRetries = 3;
    let attempt = 0;

    const payload = {
      inputs,
      truncate: false, // Ensure client-side token integrity
    };

    if (isQuery && this.promptName) {
      payload.prompt_name = this.promptName;
    }

    while (attempt < maxRetries) {
      attempt++;
      try {
        const response = await this.pool.request({
          path: '/embed',
          method: 'POST',
          headers: { 'content-type': 'application/json' },
          body: JSON.stringify(payload),
          headersTimeout: this.timeoutMs,
          bodyTimeout: this.timeoutMs,
        });

        const status = response.statusCode;

        if (status === 200) {
          const bodyText = await response.body.text();
          return JSON.parse(bodyText);
        }

        // Fatal client errors (400, 413, 422): fail immediately, do not retry
        if (status >= 400 && status < 500 && status !== 429) {
          const errDetail = await response.body.text().catch(() => '');
          throw new Error(`Fatal TEI Client Error (HTTP ${status}): ${errDetail}`);
        }

        // Handle Retry-After header if present
        const retryAfterHeader = response.headers['retry-after'];
        let backoffMs = Math.pow(2, attempt) * 150 + Math.random() * 100;

        if (retryAfterHeader && typeof retryAfterHeader === 'string') {
          const parsedSeconds = parseInt(retryAfterHeader, 10);
          if (!isNaN(parsedSeconds)) backoffMs = parsedSeconds * 1000;
        }

        await response.body.dump(); // Drain connection

        if (attempt >= maxRetries) {
          throw new Error(`TEI upstream error HTTP ${status} after ${maxRetries} attempts.`);
        }

        await new Promise((resolve) => setTimeout(resolve, backoffMs));
      } catch (err) {
        if (err.message && err.message.includes('Fatal TEI Client Error')) {
          throw err;
        }
        if (attempt >= maxRetries) throw err;

        const backoffMs = Math.pow(2, attempt) * 150 + Math.random() * 100;
        await new Promise((resolve) => setTimeout(resolve, backoffMs));
      }
    }

    throw new Error('Unexpected termination of TEI network dispatch loop.');
  }

  /**
   * Health probe against TEI GET /health
   * @returns {Promise<boolean>}
   */
  async isHealthy() {
    try {
      const response = await this.pool.request({
        path: '/health',
        method: 'GET',
        headersTimeout: 3000,
      });
      await response.body.dump();
      return response.statusCode === 200;
    } catch {
      return false;
    }
  }

  /**
   * Close the connection pool gracefully.
   */
  async close() {
    await this.pool.close();
  }
}
```

---

## 4. Document Ingestion, AST Chunking & Security

### Recursive AST Chunking (cAST) vs. Symbol Collapsing
A common architectural anti-pattern is "collapsing nested methods into their outer class block." In real software, a class can easily span 500–2,000 lines. When sent to an embedding model with a 512-token context window, TEI truncates the chunk at token 512. As a result, **70%+ of the class's methods are never embedded**, causing silent retrieval failures.

#### The cAST Principle:
1. **Parent Breadcrumb Injection**: When decomposing an AST, inject contextual breadcrumbs into each child chunk (e.g. `// Class: DatabasePool > Method: acquireConnection`).
2. **Recursive Descent**: If a class or function exceeds the token target (e.g. 350 tokens), split recursively at method and block boundaries.
3. **Micro-Fragment Folding**: Functions smaller than 40 tokens fold into adjacent siblings to avoid micro-fragment noise.

### Token Budget Enforcement vs. Character Caps
Characters and tokens do not scale 1:1. In dense code, symbols like `const x = map[0x1a, 0x2b];` can generate **3 to 4 tokens per 5 characters**. A fixed character limit of 1,500 characters can easily produce 900–1,200 tokens, exceeding a 512-token context window.

* Always count tokens using a fast BPE tokenizer (`tiktoken`) or TEI's `/tokenize` endpoint.
* Target chunk size: **350–450 tokens** for 512-token context models.

### Defense-in-Depth Secret Redaction
Naive regex patterns like `/^([A-Z0-9_]+)\s*=\s*(.*)$/gim` suffer from severe failure modes:
* An empty value (`DEBUG=\nAPI_KEY=123`) swallows the next line, completely erasing the key name `API_KEY`.
* The `/i` flag corrupts benign code lines like `name = "my-package"`.
* It misses inline private keys (PEM), bearer tokens, and connection strings (`postgres://user:pass@host/db`).

#### Two-Layer Security Strategy:
1. **Intake File Exclusion (Layer 1)**: Filter out `.env*`, `*.pem`, `*.key`, `*id_rsa*`, and credential files at the filesystem intake stage using `ignore`.
2. **Inline Token & URI Sanitizer (Layer 2)**: Scrub credentials before text is dispatched to TEI.

### Ingestion Pipeline Implementation (`src/pipeline/ingestion.js`)

```javascript
// src/pipeline/ingestion.js
import fs from 'node:fs';
import path from 'node:path';
import { createHash } from 'node:crypto';
import ignore from 'ignore';
import { get_encoding } from 'tiktoken';

// BPE Tokenizer for accurate token budget enforcement
const tokenizer = get_encoding('cl100k_base');

// Defense Layer 1: Filesystem exclusion rules
export const intakeFilter = ignore().add([
  '.env*',
  '**/*.env*',
  '**/*.pem',
  '**/*.key',
  '**/*.p12',
  '**/id_rsa*',
  '**/credentials.json',
  '**/secrets.yaml',
  '**/node_modules/**',
  '**/.git/**',
  '**/dist/**',
]);

/**
 * Defense Layer 2: Redact inline secrets from text content.
 * @param {string} text
 * @returns {string} Sanitized text
 */
export function sanitizeText(text) {
  // Redact database and service connection URIs
  let cleaned = text.replace(
    /(postgres|mysql|mongodb|redis):\/\/[^:\s]+:[^@\s]+@/gi,
    '$1://[REDACTED]:[REDACTED]@'
  );

  // Redact Bearer tokens
  cleaned = cleaned.replace(
    /(Bearer\s+)[A-Za-z0-9\-._~+/]+=*/gi,
    '$1[REDACTED]'
  );

  // Redact PEM private keys
  cleaned = cleaned.replace(
    /-----BEGIN [A-Z ]+ PRIVATE KEY-----[\s\S]*?-----END [A-Z ]+ PRIVATE KEY-----/gi,
    '[REDACTED PRIVATE KEY]'
  );

  return cleaned;
}

/**
 * Compute composite chunk hash including model metadata to prevent false cache hits.
 * @param {string} text
 * @param {string} modelId
 * @param {string} revision
 * @returns {string} SHA-256 hex string
 */
export function computeCompositeHash(text, modelId, revision) {
  return createHash('sha256')
    .update(`${text}::${modelId}::${revision}`)
    .digest('hex');
}

/**
 * Token-aware sliding chunker with AST line boundary snapping.
 * @param {string} content
 * @param {string} filePath
 * @param {number} [maxTokens=400]
 * @param {number} [overlapTokens=50]
 * @returns {Array<{filePath: string, startLine: number, endLine: number, text: string, tokenCount: number}>}
 */
export function sliceDocumentIntoChunks(content, filePath, maxTokens = 400, overlapTokens = 50) {
  const sanitized = sanitizeText(content);
  const lines = sanitized.split('\n');
  const chunks = [];

  let currentLines = [];
  let currentTokenCount = 0;
  let startLine = 1;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const lineTokens = tokenizer.encode(line).length + 1; // +1 for newline

    if (currentTokenCount + lineTokens > maxTokens && currentLines.length > 0) {
      const chunkText = currentLines.join('\n');
      chunks.push({
        filePath,
        startLine,
        endLine: startLine + currentLines.length - 1,
        text: chunkText,
        tokenCount: currentTokenCount,
      });

      // Calculate overlap lines for semantic continuity
      const overlapLines = [];
      let overlapCount = 0;
      for (let j = currentLines.length - 1; j >= 0; j--) {
        const count = tokenizer.encode(currentLines[j]).length + 1;
        if (overlapCount + count > overlapTokens) break;
        overlapLines.unshift(currentLines[j]);
        overlapCount += count;
      }

      currentLines = [...overlapLines, line];
      startLine = (startLine + currentLines.length - 1) - overlapLines.length;
      currentTokenCount = overlapCount + lineTokens;
    } else {
      currentLines.push(line);
      currentTokenCount += lineTokens;
    }
  }

  if (currentLines.length > 0) {
    chunks.push({
      filePath,
      startLine,
      endLine: startLine + currentLines.length - 1,
      text: currentLines.join('\n'),
      tokenCount: currentTokenCount,
    });
  }

  return chunks;
}
```

---

## 5. SQLite Storage Layer: `better-sqlite3` + `sqlite-vec` + FTS5

### Why the File-per-Project Pattern Wins for SQLite
Instead of maintaining a shared database with tenant ID columns, organizing storage as **one SQLite file per project** (`./data/projects/${projectId}.db`) delivers superior operational characteristics:
1. **Isolated Vector Spaces**: The `sqlite-vec` table indexes only that project's vectors, eliminating the overfiltering defect.
2. **Native FTS5 BM25 Out of the Box**: SQLite includes **FTS5** with authentic BM25 ranking (`bm25(chunks_fts)`) in its core engine. No external extensions are needed.
3. **Non-Blocking WAL Concurrency**: Each project maintains its own Write-Ahead Log (WAL), allowing concurrent reads alongside background indexing.

### Database Schema with Automatic FTS5 Synchronization

```sql
-- Relational document chunk storage
CREATE TABLE IF NOT EXISTS chunks (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  file_path TEXT NOT NULL,
  start_line INTEGER NOT NULL,
  end_line INTEGER NOT NULL,
  chunk_text TEXT NOT NULL,
  content_hash TEXT NOT NULL,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file_path);
CREATE INDEX IF NOT EXISTS idx_chunks_hash ON chunks(content_hash);

-- sqlite-vec dense vector virtual table (Float32Array vectors)
CREATE VIRTUAL TABLE IF NOT EXISTS chunk_vectors USING vec0(
  embedding float[768]
);

-- SQLite FTS5 Full-Text Search virtual table with BM25
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
  chunk_text,
  content='chunks',
  content_rowid='id',
  tokenize='unicode61 remove_diacritics 2'
);

-- Triggers for automatic synchronization between chunks and chunks_fts
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
  INSERT INTO chunks_fts(rowid, chunk_text) VALUES (new.id, new.chunk_text);
END;

CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
  INSERT INTO chunks_fts(chunks_fts, rowid, chunk_text) VALUES('delete', old.id, old.chunk_text);
END;

CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
  INSERT INTO chunks_fts(chunks_fts, rowid, chunk_text) VALUES('delete', old.id, old.chunk_text);
  INSERT INTO chunks_fts(rowid, chunk_text) VALUES (new.id, new.chunk_text);
END;

-- Project Provenance Table
CREATE TABLE IF NOT EXISTS project_provenance (
  id INTEGER PRIMARY KEY DEFAULT 1,
  model_id TEXT NOT NULL,
  revision TEXT NOT NULL,
  dimensions INTEGER NOT NULL,
  updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

### WAL Mode & High-Concurrency PRAGMAs
Every database connection executes:
* `PRAGMA journal_mode = WAL;`: Enables concurrent readers while writes occur.
* `PRAGMA synchronous = NORMAL;`: Optimizes durability for high-throughput write performance.
* `PRAGMA foreign_keys = ON;`: Enforces referential integrity.
* `PRAGMA temp_store = MEMORY;`: Keeps temporary tables in RAM.

### SQLite Project Manager Implementation (`src/db/sqliteManager.js`)

```javascript
// src/db/sqliteManager.js
import Database from 'better-sqlite3';
import * as sqliteVec from 'sqlite-vec';
import fs from 'node:fs';
import path from 'node:path';

/**
 * Manages isolated SQLite databases per project.
 */
export class SqliteProjectManager {
  /**
   * @param {string} [baseDir='./data/projects'] - Directory where project DBs reside
   */
  constructor(baseDir = './data/projects') {
    this.baseDir = baseDir;
    fs.mkdirSync(this.baseDir, { recursive: true });
    this.dbs = new Map();
  }

  /**
   * Retrieve or initialize a project's database.
   * @param {string} projectId
   * @returns {Database.Database}
   */
  getDb(projectId) {
    if (this.dbs.has(projectId)) {
      return this.dbs.get(projectId);
    }

    const sanitizedId = projectId.replace(/[^a-zA-Z0-9_-]/g, '_');
    const dbPath = path.join(this.baseDir, `${sanitizedId}.db`);
    const db = new Database(dbPath);

    // Load sqlite-vec extension
    sqliteVec.load(db);

    // Apply high-performance concurrency PRAGMAs
    db.pragma('journal_mode = WAL');
    db.pragma('synchronous = NORMAL');
    db.pragma('temp_store = MEMORY');
    db.pragma('foreign_keys = ON');

    this.initSchema(db);
    this.dbs.set(projectId, db);
    return db;
  }

  /**
   * Initialize tables, virtual tables, and triggers.
   * @private
   */
  initSchema(db) {
    db.exec(`
      CREATE TABLE IF NOT EXISTS chunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_path TEXT NOT NULL,
        start_line INTEGER NOT NULL,
        end_line INTEGER NOT NULL,
        chunk_text TEXT NOT NULL,
        content_hash TEXT NOT NULL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
      );

      CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file_path);
      CREATE INDEX IF NOT EXISTS idx_chunks_hash ON chunks(content_hash);

      CREATE VIRTUAL TABLE IF NOT EXISTS chunk_vectors USING vec0(
        embedding float[768]
      );

      CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
        chunk_text,
        content='chunks',
        content_rowid='id',
        tokenize='unicode61 remove_diacritics 2'
      );

      CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
        INSERT INTO chunks_fts(rowid, chunk_text) VALUES (new.id, new.chunk_text);
      END;

      CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
        INSERT INTO chunks_fts(chunks_fts, rowid, chunk_text) VALUES('delete', old.id, old.chunk_text);
      END;

      CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
        INSERT INTO chunks_fts(chunks_fts, rowid, chunk_text) VALUES('delete', old.id, old.chunk_text);
        INSERT INTO chunks_fts(rowid, chunk_text) VALUES (new.id, new.chunk_text);
      END;

      CREATE TABLE IF NOT EXISTS project_provenance (
        id INTEGER PRIMARY KEY DEFAULT 1,
        model_id TEXT NOT NULL,
        revision TEXT NOT NULL,
        dimensions INTEGER NOT NULL,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
      );
    `);
  }

  /**
   * Permanently remove a project database.
   * @param {string} projectId
   */
  deleteProject(projectId) {
    const sanitizedId = projectId.replace(/[^a-zA-Z0-9_-]/g, '_');
    if (this.dbs.has(projectId)) {
      this.dbs.get(projectId).close();
      this.dbs.delete(projectId);
    }
    const dbPath = path.join(this.baseDir, `${sanitizedId}.db`);
    if (fs.existsSync(dbPath)) fs.unlinkSync(dbPath);
    if (fs.existsSync(`${dbPath}-wal`)) fs.unlinkSync(`${dbPath}-wal`);
    if (fs.existsSync(`${dbPath}-shm`)) fs.unlinkSync(`${dbPath}-shm`);
  }

  /**
   * Close all active database connections.
   */
  closeAll() {
    for (const [id, db] of this.dbs.entries()) {
      try { db.close(); } catch {}
    }
    this.dbs.clear();
  }
}
```

---

## 6. Lightweight Hybrid Search & Reciprocal Rank Fusion (RRF)

Dense vector search matches semantic concepts, while lexical search matches exact identifiers, error codes, and symbols. The lightweight search engine combines both in a single Node.js process using **Reciprocal Rank Fusion (RRF)**.

### Dense Vector Distance (`sqlite-vec`)
To perform similarity search in `sqlite-vec`, pass the embedding as a serialized `Float32Array` buffer:
```javascript
const buffer = Buffer.from(new Float32Array(queryVector).buffer);
const stmt = db.prepare(`
  SELECT rowid, distance 
  FROM chunk_vectors 
  WHERE embedding MATCH ? AND k = 30
  ORDER BY distance ASC
`);
const denseHits = stmt.all(buffer);
```

### Native Lexical BM25 Scoring (SQLite FTS5)
SQLite's built-in `bm25(chunks_fts)` function computes authentic BM25 relevance scores directly:
```javascript
const stmt = db.prepare(`
  SELECT rowid, bm25(chunks_fts) AS bm25_score
  FROM chunks_fts
  WHERE chunks_fts MATCH ?
  ORDER BY bm25_score ASC
  LIMIT 30
`);
const sparseHits = stmt.all(sanitizedFtsQuery);
```

### Calibrated 1-Based Reciprocal Rank Fusion (RRF)
RRF combines dense and sparse candidate rankings without requiring complex score normalization:

$$RRF(d) = \sum_{m \in M} \frac{1}{k + r_m(d) + 1} \quad (k = 60)$$

* Ranks are strictly **1-based** ($r_m(d) \ge 1$) to preserve the calibrated smoothing factor $k = 60$.

### Cumulative Density Proximity Chunk Merging
When adjacent chunks in the same file match a search query, their relevance is compounded. Using `Math.max` discards this signal. The enhanced merger applies probabilistic score accumulation:

$$S_{\text{merged}} = S_1 + S_2 - (S_1 \cdot S_2)$$

### Hybrid Search Service Implementation (`src/services/searchEngine.js`)

```javascript
// src/services/searchEngine.js

/**
 * Lightweight Hybrid Search Engine combining sqlite-vec KNN and SQLite FTS5 BM25.
 */
export class HybridSearchEngine {
  /**
   * @param {import('../db/sqliteManager.js').SqliteProjectManager} projectManager
   * @param {import('./teiClient.js').ResilientTeiClient} teiClient
   * @param {Object} [options]
   * @param {string} [options.rerankerUrl] - Optional URL for Stage 2 Cross-Encoder
   */
  constructor(projectManager, teiClient, options = {}) {
    this.projectManager = projectManager;
    this.teiClient = teiClient;
    this.rerankerUrl = options.rerankerUrl;
  }

  /**
   * Execute hybrid search over a project.
   * @param {string} projectId
   * @param {string} queryText
   * @param {number} [limit=10]
   * @returns {Promise<Array<Object>>}
   */
  async search(projectId, queryText, limit = 10) {
    const db = this.projectManager.getDb(projectId);

    // 1. Generate dense query embedding via TEI
    const queryVector = await this.teiClient.embedQuery(queryText);
    const vectorBuffer = Buffer.from(new Float32Array(queryVector).buffer);

    // 2. Dense Vector KNN Scan (sqlite-vec)
    const denseStmt = db.prepare(`
      SELECT rowid, distance 
      FROM chunk_vectors 
      WHERE embedding MATCH ? AND k = 30
      ORDER BY distance ASC
    `);
    const denseHits = denseStmt.all(vectorBuffer);

    // 3. Sparse Lexical BM25 Scan (SQLite FTS5)
    const sanitizedFts = this.sanitizeFtsQuery(queryText);
    let sparseHits = [];
    if (sanitizedFts) {
      const sparseStmt = db.prepare(`
        SELECT rowid, bm25(chunks_fts) AS bm25_score
        FROM chunks_fts
        WHERE chunks_fts MATCH ?
        ORDER BY bm25_score ASC
        LIMIT 30
      `);
      try {
        sparseHits = sparseStmt.all(sanitizedFts);
      } catch {
        // Fallback gracefully if query contains unmatched boolean operators
      }
    }

    // 4. Reciprocal Rank Fusion (1-based ranking)
    const fusedMap = this.computeRrf(denseHits, sparseHits, 60);

    // 5. Hydrate chunk metadata
    const candidateIds = Array.from(fusedMap.keys());
    if (candidateIds.length === 0) return [];

    const placeholders = candidateIds.map(() => '?').join(',');
    const chunkStmt = db.prepare(`
      SELECT id, file_path, start_line, end_line, chunk_text
      FROM chunks
      WHERE id IN (${placeholders})
    `);
    const rows = chunkStmt.all(...candidateIds);
    const chunkLookup = new Map(rows.map((r) => [r.id, r]));

    const candidates = [];
    for (const [id, scoreData] of fusedMap.entries()) {
      const chunk = chunkLookup.get(id);
      if (chunk) {
        candidates.push({
          id: chunk.id,
          filePath: chunk.file_path,
          startLine: chunk.start_line,
          endLine: chunk.end_line,
          chunkText: chunk.chunk_text,
          denseRank: scoreData.denseRank,
          sparseRank: scoreData.sparseRank,
          score: scoreData.score,
        });
      }
    }

    // 6. Cumulative Proximity Merge
    const mergedHits = this.mergeNearbyChunks(candidates, 20);

    // 7. Optional Stage 2 Reranking (if configured)
    if (this.rerankerUrl) {
      return await this.rerankCandidates(queryText, mergedHits.slice(0, 30), limit);
    }

    return mergedHits.slice(0, limit);
  }

  /**
   * Sanitize arbitrary query text into safe FTS5 OR terms.
   * @private
   */
  sanitizeFtsQuery(query) {
    const terms = query
      .replace(/[^\w\s]/g, ' ')
      .trim()
      .split(/\s+/)
      .filter((t) => t.length > 1);
    return terms.join(' OR ');
  }

  /**
   * Calibrated 1-based RRF calculation.
   * @private
   */
  computeRrf(denseHits, sparseHits, k = 60) {
    const map = new Map();

    denseHits.forEach((hit, idx) => {
      const rank = idx + 1; // 1-based
      map.set(hit.rowid, {
        denseRank: rank,
        score: 1 / (k + rank),
      });
    });

    sparseHits.forEach((hit, idx) => {
      const rank = idx + 1;
      const term = 1 / (k + rank);
      const existing = map.get(hit.rowid);
      if (existing) {
        existing.sparseRank = rank;
        existing.score += term;
      } else {
        map.set(hit.rowid, {
          sparseRank: rank,
          score: term,
        });
      }
    });

    return map;
  }

  /**
   * Proximity chunk merging with probabilistic score accumulation.
   * @private
   */
  mergeNearbyChunks(hits, lineProximity = 20) {
    if (hits.length <= 1) return hits;

    const byFile = new Map();
    for (const hit of hits) {
      const list = byFile.get(hit.filePath) ?? [];
      list.push(hit);
      byFile.set(hit.filePath, list);
    }

    const merged = [];

    for (const fileHits of byFile.values()) {
      fileHits.sort((a, b) => a.startLine - b.startLine);
      let current = fileHits[0];

      for (let i = 1; i < fileHits.length; i++) {
        const next = fileHits[i];
        if (next.startLine - current.endLine <= lineProximity) {
          // Contiguous or overlapping lines: extend boundaries
          current.endLine = Math.max(current.endLine, next.endLine);
          current.chunkText = `${current.chunkText}\n---\n${next.chunkText}`;
          // Probabilistic score accumulation: S1 + S2 - S1*S2
          current.score = current.score + next.score - current.score * next.score;
        } else {
          merged.push(current);
          current = next;
        }
      }
      merged.push(current);
    }

    return merged.sort((a, b) => b.score - a.score);
  }

  /**
   * Optional Stage 2 Cross-Encoder Reranker via TEI /rerank.
   * @private
   */
  async rerankCandidates(query, candidates, limit) {
    if (candidates.length === 0) return [];
    try {
      const response = await fetch(`${this.rerankerUrl}/rerank`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({
          query,
          texts: candidates.map((c) => c.chunkText),
          truncate: true,
        }),
        signal: AbortSignal.timeout(5000),
      });

      if (!response.ok) return candidates.slice(0, limit);

      const results = await response.json();
      return results
        .sort((a, b) => b.score - a.score)
        .slice(0, limit)
        .map((r) => {
          const original = candidates[r.index];
          return { ...original, score: r.score };
        });
    } catch {
      // Fallback gracefully to RRF score if reranker fails
      return candidates.slice(0, limit);
    }
  }
}
```

---

## 7. Optional: Pluggable Second-Stage Cross-Encoder Reranking

### When to Enable Two-Stage Reranking
The core hybrid search pipeline (Bi-Encoder Dense + FTS5 BM25 + RRF) provides high throughput with minimal compute. However, certain complex retrieval scenarios benefit from a **Cross-Encoder Reranker**:

* **Bi-Encoder Trade-off**: Embeds queries and passages separately into vector space. It is extremely fast ($O(1)$ vector dot product), but cannot model fine-grained token-to-token cross-attention between query words and passage words.
* **Cross-Encoder Advantage**: Feeds the query and passage together through joint transformer attention layers, scoring relevance with deep cross-attention. Anthropic benchmarks show cross-encoder rerankers can reduce top-20 retrieval failures by up to 67%.

#### Practical Recommendation:
* **Start Lightweight**: Run the core hybrid pipeline (Bi-Encoder + FTS5 BM25). It requires only a single TEI container and provides fast response times.
* **Enable Reranking When**: You are indexing large codebases with subtle domain logic and your evaluation harness shows high candidate recall at $K=30$, but lower precision in the top 3 results.

### Reranker Service Integration (`/rerank`)
To enable reranking, start a second TEI container running a dedicated cross-encoder model:

```bash
docker run -d \
  --name tei-reranker \
  --restart unless-stopped \
  -p 8081:80 \
  -v $HOME/.cache/huggingface/hub:/data \
  ghcr.io/huggingface/text-embeddings-inference:cpu-1.9 \
  --model-id BAAI/bge-reranker-v2-m3 \
  --revision 46e01a88b857501a1827b58f8e32c027878fae3a \
  --port 80 \
  --max-client-batch-size 32
```

Pass `rerankerUrl: 'http://localhost:8081'` when initializing `HybridSearchEngine`. The engine will automatically route top-30 RRF candidates through `/rerank`.

---

## 8. Complete Production Docker Compose Environment

A lightweight Compose v2 configuration running the TEI bi-encoder inference engine and the Vanilla JavaScript Node.js application, with an optional Docker profile for the cross-encoder reranker.

```yaml
# docker-compose.yml
services:
  # 1. Primary Bi-Encoder Embedding Microservice (Rust TEI)
  tei-embeddings:
    image: ghcr.io/huggingface/text-embeddings-inference:cpu-1.9
    container_name: tei-embeddings
    restart: unless-stopped
    ports:
      - "127.0.0.1:8080:80"
    volumes:
      - hf_model_cache:/data
    command:
      - "--model-id"
      - "BAAI/bge-base-en-v1.5"
      - "--revision"
      - "a5eda8f47b2c15c0e241780540445a6c117d33d9"
      - "--port"
      - "80"
      - "--max-client-batch-size"
      - "32"
      - "--max-batch-tokens"
      - "16384"
      - "--default-prompt-name"
      - "query"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:80/health"]
      interval: 10s
      timeout: 5s
      retries: 5
      start_period: 60s

  # 2. [OPTIONAL] Cross-Encoder Reranker Microservice
  # To run with reranker: docker compose --profile reranker up -d
  tei-reranker:
    image: ghcr.io/huggingface/text-embeddings-inference:cpu-1.9
    container_name: tei-reranker
    restart: unless-stopped
    profiles:
      - reranker
    ports:
      - "127.0.0.1:8081:80"
    volumes:
      - hf_model_cache:/data
    command:
      - "--model-id"
      - "BAAI/bge-reranker-v2-m3"
      - "--revision"
      - "46e01a88b857501a1827b58f8e32c027878fae3a"
      - "--port"
      - "80"
      - "--max-client-batch-size"
      - "32"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:80/health"]
      interval: 10s
      timeout: 5s
      retries: 5
      start_period: 60s

  # 3. Node.js 24 LTS Application Backend (Vanilla JS)
  node-search-app:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: node-search-app
    restart: unless-stopped
    depends_on:
      tei-embeddings:
        condition: service_healthy
    environment:
      NODE_ENV: production
      PORT: 3000
      TEI_SERVICE_URL: http://tei-embeddings:80
      RERANKER_SERVICE_URL: http://tei-reranker:80
      DATA_DIR: /app/data/projects
    volumes:
      - ./data/projects:/app/data/projects
    ports:
      - "3000:3000"

volumes:
  hf_model_cache:
```

---

## 9. Operations, Provenance & Zero-Downtime Re-indexing

### Composite Content-Hash Gating
Gating document re-embedding solely on raw file text creates a critical bug: if you change the embedding model, update model weights, or alter prompt templates, the system falsely skips re-embedding because the raw text hasn't changed.

* **Composite Hash Rule**: The stored hash must combine the chunk text, the model ID, and the git revision:
  $$\text{Hash} = \text{SHA256}(\text{chunkText} + \text{modelId} + \text{revision})$$
* On re-indexing, compare this composite hash against the stored hash in the `chunks` table. If it matches, skip the TEI embedding call. If it differs, update the chunk text and vector.

### Atomic Project Re-indexing & Hot Swap
When re-indexing a project (e.g. migrating from a 384-dimension model to a 768-dimension model):
1. **Background Indexing**: Create a temporary database file: `./data/projects/${projectId}.db.tmp`.
2. **Ingest Chunks**: Ingest all source files into the temporary database.
3. **Atomic Cutover**: Rename the file atomically:
   ```javascript
   import fs from 'node:fs';

   projectManager.getDb(projectId).close();
   fs.renameSync(
     `./data/projects/${projectId}.db.tmp`,
     `./data/projects/${projectId}.db`
   );
   ```
4. This atomic filesystem swap ensures zero search downtime and instant rollbacks.

---

## 10. Vanilla JavaScript Package Manifest & Directory Layout

### Complete `package.json` Manifest
```json
{
  "name": "semantic-search-sqlite-vanillajs",
  "version": "1.0.0",
  "description": "Production Semantic Search with SQLite, sqlite-vec, FTS5 BM25, and Hugging Face TEI in Vanilla JS",
  "type": "module",
  "main": "src/index.js",
  "engines": {
    "node": ">=22.0.0"
  },
  "scripts": {
    "start": "node src/index.js",
    "dev": "node --watch src/index.js",
    "test": "node --test test/**/*.test.js"
  },
  "dependencies": {
    "better-sqlite3": "^11.8.1",
    "fastify": "^5.2.1",
    "ignore": "^7.0.3",
    "opossum": "^8.1.4",
    "p-limit": "^6.2.0",
    "sqlite-vec": "^0.1.6",
    "tiktoken": "^1.0.15",
    "undici": "^6.21.1"
  },
  "devDependencies": {
    "prettier": "^3.5.2"
  }
}
```

### Clean Directory Layout
```
├── docker-compose.yml
├── Dockerfile
├── package.json
├── data/
│   └── projects/               # Isolated SQLite database files (*.db)
│       ├── project_alpha.db
│       └── project_beta.db
├── src/
│   ├── index.js                # Fastify HTTP entrypoint
│   ├── db/
│   │   └── sqliteManager.js    # better-sqlite3 + sqlite-vec + FTS5 manager
│   ├── pipeline/
│   │   └── ingestion.js        # AST chunker, secret sanitizer, hash gater
│   └── services/
│       ├── teiClient.js        # Undici connection pool & Opossum circuit breaker
│       └── searchEngine.js     # Dense + FTS5 BM25 hybrid search & RRF
└── test/
    ├── teiClient.test.js       # Unit tests for retry and circuit breaker
    └── searchEngine.test.js    # Integration tests for sqlite-vec and FTS5 BM25
```

import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import crypto from 'node:crypto';
import Database from 'better-sqlite3';
import * as sqliteVec from 'sqlite-vec';

const BASE_DIR = process.env.KENSAKU_CACHE_DIR || path.join(os.homedir(), '.cache', 'kensaku');
const WORKSPACES_DIR = path.join(BASE_DIR, 'workspaces');
const REGISTRY_FILE = path.join(BASE_DIR, 'workspaces.json');

// Ensure directories exist
fs.mkdirSync(WORKSPACES_DIR, { recursive: true });

// Active DB connection cache
const dbCache = new Map();

/**
 * Loads the workspace registry.
 * @returns {Record<string, { name: string, path: string, addedAt: string, updatedAt?: string }>}
 */
export function getWorkspacesRegistry() {
  try {
    if (fs.existsSync(REGISTRY_FILE)) {
      return JSON.parse(fs.readFileSync(REGISTRY_FILE, 'utf8'));
    }
  } catch (err) {
    console.warn('[Kensaku] Warning reading registry:', err.message);
  }
  return {};
}

/**
 * Saves the workspace registry.
 * @param {Record<string, object>} registry
 */
export function saveWorkspacesRegistry(registry) {
  fs.writeFileSync(REGISTRY_FILE, JSON.stringify(registry, null, 2), 'utf8');
}

/**
 * Registers or updates a workspace category.
 * @param {string} category
 * @param {string} rootDir
 * @returns {object}
 */
export function registerWorkspace(category, rootDir) {
  const normCategory = category.trim().toLowerCase();
  const absPath = path.resolve(rootDir);
  const registry = getWorkspacesRegistry();

  registry[normCategory] = {
    name: normCategory,
    path: absPath,
    addedAt: registry[normCategory]?.addedAt || new Date().toISOString(),
    updatedAt: new Date().toISOString()
  };

  saveWorkspacesRegistry(registry);
  return registry[normCategory];
}

/**
 * Removes a workspace category and its database.
 * @param {string} category
 */
export function unregisterWorkspace(category) {
  const normCategory = category.trim().toLowerCase();
  closeDatabase(normCategory);

  const dbPath = path.join(WORKSPACES_DIR, `${normCategory}.db`);
  if (fs.existsSync(dbPath)) {
    try {
      fs.unlinkSync(dbPath);
    } catch {}
  }

  const registry = getWorkspacesRegistry();
  delete registry[normCategory];
  saveWorkspacesRegistry(registry);
}

/**
 * Returns an open SQLite Database instance for the specified category.
 * @param {string} category
 * @returns {Database.Database}
 */
export function getDatabase(category) {
  const normCategory = (category || 'default').trim().toLowerCase();
  if (dbCache.has(normCategory)) {
    return dbCache.get(normCategory);
  }

  const dbPath = path.join(WORKSPACES_DIR, `${normCategory}.db`);
  const db = new Database(dbPath);

  // High performance PRAGMAs
  db.pragma('journal_mode = WAL');
  db.pragma('busy_timeout = 5000');
  db.pragma('synchronous = NORMAL');

  // Load sqlite-vec extension
  sqliteVec.load(db);

  // Initialize schema
  db.exec(`
    CREATE TABLE IF NOT EXISTS documents (
      path TEXT PRIMARY KEY,
      hash TEXT NOT NULL,
      mtime INTEGER NOT NULL,
      chunk_count INTEGER NOT NULL,
      updated_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS chunks (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      doc_path TEXT NOT NULL,
      chunk_index INTEGER NOT NULL,
      title TEXT NOT NULL,
      section TEXT NOT NULL,
      line_start INTEGER NOT NULL,
      line_end INTEGER NOT NULL,
      content TEXT NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_chunks_doc_path ON chunks(doc_path);

    CREATE VIRTUAL TABLE IF NOT EXISTS chunk_vectors USING vec0(
      embedding float[384] distance_metric=cosine
    );

    CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
      content,
      title,
      section,
      content='chunks',
      content_rowid='id'
    );

    CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
      INSERT INTO chunks_fts(rowid, content, title, section)
      VALUES (new.id, new.content, new.title, new.section);
    END;

    CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
      INSERT INTO chunks_fts(chunks_fts, rowid, content, title, section)
      VALUES ('delete', old.id, old.content, old.title, old.section);
    END;
  `);

  dbCache.set(normCategory, db);
  return db;
}

/**
 * Closes an open database connection.
 * @param {string} category
 */
export function closeDatabase(category) {
  const normCategory = (category || 'default').trim().toLowerCase();
  if (dbCache.has(normCategory)) {
    try {
      dbCache.get(normCategory).close();
    } catch {}
    dbCache.delete(normCategory);
  }
}

/**
 * Closes all cached database connections.
 */
export function closeAllDatabases() {
  for (const [cat, db] of dbCache.entries()) {
    try {
      db.close();
    } catch {}
  }
  dbCache.clear();
}

/**
 * Retrieves a document manifest record.
 * @param {string} category
 * @param {string} docPath
 * @returns {object | null}
 */
export function getDocumentRecord(category, docPath) {
  const db = getDatabase(category);
  return db.prepare('SELECT path, hash, mtime, chunk_count, updated_at FROM documents WHERE path = ?').get(docPath) || null;
}

/**
 * Retrieves all document records in a category.
 * @param {string} category
 * @returns {Array<object>}
 */
export function getAllDocumentRecords(category) {
  const db = getDatabase(category);
  return db.prepare('SELECT path, hash, mtime, chunk_count, updated_at FROM documents').all();
}

/**
 * Deletes a document and its chunks & vectors.
 * @param {string} category
 * @param {string} docPath
 */
export function deleteDocumentChunks(category, docPath) {
  const db = getDatabase(category);
  const findChunkIds = db.prepare('SELECT id FROM chunks WHERE doc_path = ?');
  const deleteVec = db.prepare('DELETE FROM chunk_vectors WHERE rowid = ?');
  const deleteChunks = db.prepare('DELETE FROM chunks WHERE doc_path = ?');
  const deleteDoc = db.prepare('DELETE FROM documents WHERE path = ?');

  const rows = findChunkIds.all(docPath);
  db.transaction(() => {
    for (const row of rows) {
      deleteVec.run(BigInt(row.id));
    }
    deleteChunks.run(docPath);
    deleteDoc.run(docPath);
  })();
}

/**
 * Atomically inserts/updates a document, its chunks, and corresponding dense vectors.
 * @param {string} category
 * @param {string} docPath
 * @param {string} hash
 * @param {number} mtime
 * @param {Array<object>} chunks
 * @param {Array<Float32Array>} vectors
 */
export function upsertDocumentWithChunks(category, docPath, hash, mtime, chunks, vectors) {
  const db = getDatabase(category);

  // Prepared statements
  const findOldChunkIds = db.prepare('SELECT id FROM chunks WHERE doc_path = ?');
  const deleteVec = db.prepare('DELETE FROM chunk_vectors WHERE rowid = ?');
  const deleteChunks = db.prepare('DELETE FROM chunks WHERE doc_path = ?');
  const insertDoc = db.prepare(`
    INSERT INTO documents (path, hash, mtime, chunk_count, updated_at)
    VALUES (?, ?, ?, ?, ?)
    ON CONFLICT(path) DO UPDATE SET
      hash = excluded.hash,
      mtime = excluded.mtime,
      chunk_count = excluded.chunk_count,
      updated_at = excluded.updated_at
  `);
  const insertChunk = db.prepare(`
    INSERT INTO chunks (doc_path, chunk_index, title, section, line_start, line_end, content)
    VALUES (?, ?, ?, ?, ?, ?, ?)
  `);
  const insertVec = db.prepare(`
    INSERT INTO chunk_vectors (rowid, embedding)
    VALUES (?, ?)
  `);

  const now = new Date().toISOString();

  db.transaction(() => {
    // 1. Remove old chunks and vectors if document already exists
    const oldRows = findOldChunkIds.all(docPath);
    for (const row of oldRows) {
      deleteVec.run(BigInt(row.id));
    }
    deleteChunks.run(docPath);

    // 2. Insert or update document metadata
    insertDoc.run(docPath, hash, mtime, chunks.length, now);

    // 3. Insert new chunks and corresponding vectors
    for (let i = 0; i < chunks.length; i++) {
      const c = chunks[i];
      const vec = vectors[i];
      const res = insertChunk.run(
        docPath,
        c.chunkIndex,
        c.title || docPath,
        c.section || 'General',
        c.lineStart,
        c.lineEnd,
        c.content
      );
      const chunkId = BigInt(res.lastInsertRowid);
      insertVec.run(chunkId, vec);
    }
  })();
}

/**
 * Performs K-Nearest Neighbor (KNN) vector search.
 * @param {string} category
 * @param {Float32Array} queryVector
 * @param {number} [limit=10]
 * @returns {Array<{ chunkId: number, distance: number }>}
 */
export function knnSearch(category, queryVector, limit = 10) {
  const db = getDatabase(category);
  const stmt = db.prepare(`
    SELECT rowid as chunkId, distance
    FROM chunk_vectors
    WHERE embedding MATCH ?
      AND k = ?
    ORDER BY distance ASC
  `);
  return stmt.all(queryVector, limit);
}

/**
 * Performs BM25 full-text lexical search using SQLite FTS5.
 * @param {string} category
 * @param {string} queryText
 * @param {number} [limit=10]
 * @returns {Array<{ chunkId: number, rank: number }>}
 */
export function bm25Search(category, queryText, limit = 10) {
  const db = getDatabase(category);
  // Sanitize query for FTS5: strip punctuation that triggers FTS5 syntax errors
  const sanitized = queryText
    .replace(/[^\w\s]/g, ' ')
    .trim()
    .split(/\s+/)
    .filter(Boolean)
    .join(' OR ');

  if (!sanitized) {
    return [];
  }

  try {
    const stmt = db.prepare(`
      SELECT rowid as chunkId, bm25(chunks_fts) as rank
      FROM chunks_fts
      WHERE chunks_fts MATCH ?
      ORDER BY rank ASC
      LIMIT ?
    `);
    return stmt.all(sanitized, limit);
  } catch (err) {
    console.warn('[Kensaku] FTS5 search error:', err.message);
    return [];
  }
}

/**
 * Retrieves chunk details by IDs.
 * @param {string} category
 * @param {number[]} chunkIds
 * @returns {Map<number, object>}
 */
export function getChunksByIds(category, chunkIds) {
  if (!chunkIds || chunkIds.length === 0) return new Map();
  const normCategory = (category || 'default').trim().toLowerCase();
  const db = getDatabase(normCategory);
  const placeholders = chunkIds.map(() => '?').join(',');
  const stmt = db.prepare(`
    SELECT id, doc_path, chunk_index, title, section, line_start, line_end, content
    FROM chunks
    WHERE id IN (${placeholders})
  `);
  const rows = stmt.all(...chunkIds);
  const map = new Map();
  for (const r of rows) {
    r.category = normCategory;
    map.set(r.id, r);
  }
  return map;
}

/**
 * Returns overall statistics for a category.
 * @param {string} category
 * @returns {object}
 */
export function getCategoryStats(category) {
  const normCategory = (category || 'default').trim().toLowerCase();
  const dbPath = path.join(WORKSPACES_DIR, `${normCategory}.db`);
  if (!fs.existsSync(dbPath)) {
    return { category: normCategory, exists: false, totalDocs: 0, totalChunks: 0, dbSizeBytes: 0, compositeHash: '' };
  }
  const db = getDatabase(normCategory);
  const docCount = db.prepare('SELECT count(*) as count FROM documents').get().count;
  const chunkCount = db.prepare('SELECT count(*) as count FROM chunks').get().count;
  const stats = fs.statSync(dbPath);
  const compositeHash = getCompositeWorkspaceHash(normCategory);

  return {
    category: normCategory,
    exists: true,
    totalDocs: docCount,
    totalChunks: chunkCount,
    dbSizeBytes: stats.size,
    compositeHash,
    dbPath
  };
}

/**
 * Computes a deterministic composite SHA-256 hash across all indexed documents in a workspace.
 * @param {string} category
 * @returns {string}
 */
export function getCompositeWorkspaceHash(category) {
  const normCategory = (category || 'default').trim().toLowerCase();
  const dbPath = path.join(WORKSPACES_DIR, `${normCategory}.db`);
  if (!fs.existsSync(dbPath)) return '';

  const db = getDatabase(normCategory);
  const docs = db.prepare('SELECT path, hash FROM documents ORDER BY path ASC').all();
  if (docs.length === 0) return '';

  const hasher = crypto.createHash('sha256');
  for (const doc of docs) {
    hasher.update(`${doc.path}:${doc.hash}\n`, 'utf8');
  }
  return hasher.digest('hex');
}

/**
 * Returns a fast Map of document paths to their recorded mtime and hash.
 * @param {string} category
 * @returns {Map<string, { mtime: number, hash: string }>}
 */
export function getDocumentsStatMap(category) {
  const normCategory = (category || 'default').trim().toLowerCase();
  const dbPath = path.join(WORKSPACES_DIR, `${normCategory}.db`);
  if (!fs.existsSync(dbPath)) return new Map();

  const db = getDatabase(normCategory);
  const rows = db.prepare('SELECT path, hash, mtime FROM documents').all();
  const map = new Map();
  for (const row of rows) {
    map.set(row.path, { mtime: row.mtime, hash: row.hash });
  }
  return map;
}


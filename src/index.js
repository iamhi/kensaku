import fs from 'node:fs/promises';
import { existsSync } from 'node:fs';
import path from 'node:path';
import ignore from 'ignore';
import { chunkMarkdown, hashContent } from './pipeline/chunker.js';
import { embedBatch } from './pipeline/embeddings.js';
import {
  registerWorkspace,
  unregisterWorkspace,
  getWorkspacesRegistry,
  getDocumentRecord,
  getAllDocumentRecords,
  deleteDocumentChunks,
  upsertDocumentWithChunks,
  getCategoryStats,
  closeDatabase,
  closeAllDatabases
} from './db/sqliteManager.js';
import { hybridSearch } from './services/searchEngine.js';

/**
 * Recursively scans directory for markdown files.
 * @param {string} rootDir
 * @param {string} currentDir
 * @param {import('ignore').Ignore} ig
 * @returns {Promise<string[]>} List of paths relative to rootDir
 */
async function scanMarkdownFiles(rootDir, currentDir = rootDir, ig = null) {
  const entries = await fs.readdir(currentDir, { withFileTypes: true });
  const results = [];

  for (const entry of entries) {
    const fullPath = path.join(currentDir, entry.name);
    const relPath = path.relative(rootDir, fullPath);

    // Filter out common unwanted directories and hidden files
    if (entry.name.startsWith('.') || entry.name === 'node_modules') {
      continue;
    }

    if (ig && ig.ignores(relPath)) {
      continue;
    }

    if (entry.isDirectory()) {
      const subResults = await scanMarkdownFiles(rootDir, fullPath, ig);
      results.push(...subResults);
    } else if (entry.isFile()) {
      const ext = path.extname(entry.name).toLowerCase();
      if (ext === '.md' || ext === '.markdown') {
        results.push(relPath);
      }
    }
  }

  return results;
}

/**
 * Initializes ignore patterns from .gitignore if present.
 * @param {string} rootDir
 * @returns {Promise<import('ignore').Ignore | null>}
 */
async function loadGitIgnore(rootDir) {
  const gitignorePath = path.join(rootDir, '.gitignore');
  if (existsSync(gitignorePath)) {
    try {
      const content = await fs.readFile(gitignorePath, 'utf8');
      const ig = ignore();
      ig.add(content);
      return ig;
    } catch {}
  }
  return null;
}

/**
 * Registers a workspace category with its directory.
 * @param {string} category
 * @param {string} rootDir
 * @returns {object}
 */
export function addWorkspace(category, rootDir) {
  return registerWorkspace(category, rootDir);
}

/**
 * Removes a workspace category and its index.
 * @param {string} category
 */
export function removeWorkspace(category) {
  return unregisterWorkspace(category);
}

/**
 * Lists all registered workspaces and their current stats.
 * @returns {Array<object>}
 */
export function listWorkspaces() {
  const registry = getWorkspacesRegistry();
  const list = [];
  for (const [key, item] of Object.entries(registry)) {
    const stats = getCategoryStats(key);
    list.push({
      category: key,
      path: item.path,
      addedAt: item.addedAt,
      updatedAt: item.updatedAt,
      ...stats
    });
  }
  return list;
}

/**
 * Indexes a workspace category with SHA-256 incremental hashing.
 * Only modified or new files are chunked and embedded.
 * @param {string} category - Workspace category name.
 * @param {object} [options]
 * @param {string} [options.rootDir] - Override path to workspace root.
 * @param {boolean} [options.force=false] - Force re-indexing of all files.
 * @param {number} [options.concurrency=8] - Batch embedding concurrency.
 * @returns {Promise<{ category: string, scanned: number, updated: number, unchanged: number, deleted: number, totalChunks: number, durationMs: number }>}
 */
export async function indexWorkspace(category, options = {}) {
  const startTime = Date.now();
  const normCategory = (category || 'default').trim().toLowerCase();
  const registry = getWorkspacesRegistry();
  const rootDir = options.rootDir || registry[normCategory]?.path;

  if (!rootDir || !existsSync(rootDir)) {
    throw new Error(`Workspace path does not exist for category '${normCategory}': ${rootDir}`);
  }

  // Ensure workspace is registered
  registerWorkspace(normCategory, rootDir);

  const ig = await loadGitIgnore(rootDir);
  const discoveredFiles = await scanMarkdownFiles(rootDir, rootDir, ig);
  const existingRecords = getAllDocumentRecords(normCategory);
  const existingMap = new Map(existingRecords.map((r) => [r.path, r]));

  let updatedCount = 0;
  let unchangedCount = 0;

  for (const relPath of discoveredFiles) {
    const fullPath = path.join(rootDir, relPath);
    let content = '';
    let stat;

    try {
      [content, stat] = await Promise.all([
        fs.readFile(fullPath, 'utf8'),
        fs.stat(fullPath)
      ]);
    } catch (err) {
      console.warn(`[Kensaku] Could not read ${relPath}:`, err.message);
      continue;
    }

    const currentHash = hashContent(content);
    const existing = existingMap.get(relPath);

    // Incremental skip check: same hash and mtime
    if (!options.force && existing && existing.hash === currentHash) {
      unchangedCount++;
      continue;
    }

    // Chunk the document
    const chunks = chunkMarkdown(content, relPath);
    if (chunks.length === 0) {
      deleteDocumentChunks(normCategory, relPath);
      continue;
    }

    // Embed all chunks for this document in batch
    const textsToEmbed = chunks.map((c) => `${c.title} > ${c.section}\n${c.content}`);
    const vectors = await embedBatch(textsToEmbed, { concurrency: options.concurrency || 8 });

    // Atomic transaction upsert into SQLite + sqlite-vec + FTS5
    upsertDocumentWithChunks(normCategory, relPath, currentHash, stat.mtimeMs, chunks, vectors);
    updatedCount++;
  }

  // Detect and purge deleted files
  const discoveredSet = new Set(discoveredFiles);
  let deletedCount = 0;
  for (const [recordedPath] of existingMap.entries()) {
    if (!discoveredSet.has(recordedPath)) {
      deleteDocumentChunks(normCategory, recordedPath);
      deletedCount++;
    }
  }

  const finalStats = getCategoryStats(normCategory);
  const durationMs = Date.now() - startTime;

  return {
    category: normCategory,
    rootDir,
    scanned: discoveredFiles.length,
    updated: updatedCount,
    unchanged: unchangedCount,
    deleted: deletedCount,
    totalChunks: finalStats.totalChunks,
    durationMs
  };
}

/**
 * Searches a workspace category using hybrid dense + sparse search with RRF.
 * @param {string} category
 * @param {string} query
 * @param {object} [options]
 * @returns {Promise<{ query: string, category: string, count: number, results: Array<object> }>}
 */
export async function searchWorkspace(category, query, options = {}) {
  const normCategory = (category || 'default').trim().toLowerCase();
  return hybridSearch(normCategory, query, options);
}

/**
 * Returns current status and statistics for a workspace.
 * @param {string} category
 * @returns {object}
 */
export function getWorkspaceStatus(category) {
  const normCategory = (category || 'default').trim().toLowerCase();
  const registry = getWorkspacesRegistry();
  const registered = registry[normCategory] || null;
  const stats = getCategoryStats(normCategory);

  return {
    category: normCategory,
    registered: !!registered,
    path: registered?.path || null,
    addedAt: registered?.addedAt || null,
    ...stats
  };
}

export { closeDatabase, closeAllDatabases };

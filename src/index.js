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
  getDocumentsStatMap,
  getCompositeWorkspaceHash,
  deleteDocumentChunks,
  upsertDocumentWithChunks,
  getCategoryStats,
  closeDatabase,
  closeAllDatabases
} from './db/sqliteManager.js';
import { hybridSearch, searchMultipleWorkspaces } from './services/searchEngine.js';
import { executeRAG } from './services/ragEngine.js';
import { checkLLMHealth } from './services/llmClient.js';

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
 * Quickly checks if a workspace has modified, added, or deleted files without embedding.
 * Uses mtime and file listing comparison for sub-millisecond detection.
 * @param {string} category
 * @returns {Promise<{ isStale: boolean, newFiles: string[], modifiedFiles: string[], deletedFiles: string[], compositeHash: string }>}
 */
export async function checkWorkspaceStaleness(category) {
  const normCategory = (category || 'default').trim().toLowerCase();
  const registry = getWorkspacesRegistry();
  const rootDir = registry[normCategory]?.path;

  if (!rootDir || !existsSync(rootDir)) {
    return { isStale: false, newFiles: [], modifiedFiles: [], deletedFiles: [], compositeHash: '' };
  }

  const ig = await loadGitIgnore(rootDir);
  const discoveredFiles = await scanMarkdownFiles(rootDir, rootDir, ig);
  const statMap = getDocumentsStatMap(normCategory);

  const newFiles = [];
  const modifiedFiles = [];
  const discoveredSet = new Set(discoveredFiles);

  for (const relPath of discoveredFiles) {
    const fullPath = path.join(rootDir, relPath);
    try {
      const stat = await fs.stat(fullPath);
      const existing = statMap.get(relPath);
      if (!existing) {
        newFiles.push(relPath);
      } else if (Math.floor(stat.mtimeMs) !== Math.floor(existing.mtime)) {
        modifiedFiles.push(relPath);
      }
    } catch {}
  }

  const deletedFiles = [];
  for (const recordedPath of statMap.keys()) {
    if (!discoveredSet.has(recordedPath)) {
      deletedFiles.push(recordedPath);
    }
  }

  const isStale = newFiles.length > 0 || modifiedFiles.length > 0 || deletedFiles.length > 0;
  const compositeHash = getCompositeWorkspaceHash(normCategory);

  return {
    isStale,
    newFiles,
    modifiedFiles,
    deletedFiles,
    compositeHash
  };
}

/**
 * Indexes a workspace category with SHA-256 incremental hashing.
 * Only modified or new files are chunked and embedded.
 * @param {string} category - Workspace category name.
 * @param {object} [options]
 * @param {string} [options.rootDir] - Override path to workspace root.
 * @param {boolean} [options.force=false] - Force re-indexing of all files.
 * @param {number} [options.concurrency=8] - Batch embedding concurrency.
 * @returns {Promise<{ category: string, scanned: number, updated: number, unchanged: number, deleted: number, totalChunks: number, compositeHash: string, durationMs: number }>}
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
    upsertDocumentWithChunks(normCategory, relPath, currentHash, Math.floor(stat.mtimeMs), chunks, vectors);
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
    compositeHash: finalStats.compositeHash,
    durationMs
  };
}

/**
 * Searches a single workspace or all registered workspaces using hybrid dense + sparse search with RRF.
 * If category is 'all' or empty, searches across all registered workspaces.
 * @param {string} [category='all']
 * @param {string} query
 * @param {object} [options]
 * @param {boolean} [options.autoSync=false] - Check and update index if files changed.
 * @returns {Promise<{ query: string, category: string, count: number, results: Array<object> }>}
 */
export async function searchWorkspace(category = 'all', query, options = {}) {
  const normCategory = (category || 'all').trim().toLowerCase();

  // Handle multi-workspace search across all registered workspaces
  if (normCategory === 'all') {
    const registry = getWorkspacesRegistry();
    const categories = Object.keys(registry);
    if (categories.length === 0) {
      return { query, category: 'all', count: 0, results: [] };
    }

    if (options.autoSync) {
      for (const cat of categories) {
        const { isStale } = await checkWorkspaceStaleness(cat);
        if (isStale) await indexWorkspace(cat);
      }
    }

    return searchMultipleWorkspaces(categories, query, options);
  }

  // Single workspace search
  if (options.autoSync) {
    const { isStale } = await checkWorkspaceStaleness(normCategory);
    if (isStale) await indexWorkspace(normCategory);
  }

  return hybridSearch(normCategory, query, options);
}

/**
 * Executes full RAG (Retrieval-Augmented Generation) on a query.
 * @param {string} [category='all'] - Workspace category name or 'all'.
 * @param {string} query - User question.
 * @param {object} [options]
 * @returns {Promise<{ answer: string, query: string, category: string, sources: Array<object>, durationMs: number }>}
 */
export async function askWorkspace(category = 'all', query, options = {}) {
  const normCategory = (category || 'all').trim().toLowerCase();

  if (options.autoSync) {
    if (normCategory === 'all') {
      const registry = getWorkspacesRegistry();
      for (const cat of Object.keys(registry)) {
        const { isStale } = await checkWorkspaceStaleness(cat);
        if (isStale) await indexWorkspace(cat);
      }
    } else {
      const { isStale } = await checkWorkspaceStaleness(normCategory);
      if (isStale) await indexWorkspace(normCategory);
    }
  }

  return executeRAG({
    query,
    category: normCategory,
    limit: options.limit || 5,
    llmOptions: options.llmOptions || options,
    onToken: options.onToken
  });
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

export {
  closeDatabase,
  closeAllDatabases,
  getCompositeWorkspaceHash,
  checkLLMHealth,
  executeRAG
};

import { embedText } from '../pipeline/embeddings.js';
import { knnSearch, bm25Search, getChunksByIds } from '../db/sqliteManager.js';

const RRF_K = 60; // Standard Reciprocal Rank Fusion constant

/**
 * Executes a hybrid (Dense Vector KNN + Sparse FTS5 BM25) search with Reciprocal Rank Fusion (RRF).
 * @param {string} category - Workspace category name.
 * @param {string} query - Search query text.
 * @param {object} [options]
 * @param {number} [options.limit=10] - Number of final chunks to return.
 * @param {number} [options.candidateLimit=30] - Candidate pool size for fusion.
 * @returns {Promise<{ query: string, category: string, count: number, results: Array<object> }>}
 */
export async function hybridSearch(category, query, options = {}) {
  const limit = options.limit || 10;
  const candidateLimit = options.candidateLimit || Math.max(limit * 3, 30);
  const cleanQuery = (query || '').trim();

  if (!cleanQuery) {
    return { query: '', category, count: 0, results: [] };
  }

  // 1. Generate query embedding vector
  const queryVector = await embedText(cleanQuery);

  // 2. Parallel Dense KNN + Sparse BM25 Search
  const knnResults = knnSearch(category, queryVector, candidateLimit);
  const bm25Results = bm25Search(category, cleanQuery, candidateLimit);

  // 3. Reciprocal Rank Fusion (RRF)
  const scores = new Map(); // chunkId -> { rrfScore, denseScore, bm25Score, denseRank, bm25Rank }

  // Process Dense KNN ranks
  knnResults.forEach((item, index) => {
    const rank = index + 1;
    // Cosine distance to similarity: 1 - distance
    const similarity = Math.max(0, 1 - (item.distance || 0));
    const rrfContrib = 1.0 / (RRF_K + rank);

    scores.set(item.chunkId, {
      chunkId: item.chunkId,
      rrfScore: rrfContrib,
      denseScore: similarity,
      bm25Score: 0,
      denseRank: rank,
      bm25Rank: null
    });
  });

  // Process BM25 ranks
  bm25Results.forEach((item, index) => {
    const rank = index + 1;
    const rrfContrib = 1.0 / (RRF_K + rank);

    if (scores.has(item.chunkId)) {
      const entry = scores.get(item.chunkId);
      entry.rrfScore += rrfContrib;
      entry.bm25Score = item.rank;
      entry.bm25Rank = rank;
    } else {
      scores.set(item.chunkId, {
        chunkId: item.chunkId,
        rrfScore: rrfContrib,
        denseScore: 0,
        bm25Score: item.rank,
        denseRank: null,
        bm25Rank: rank
      });
    }
  });

  if (scores.size === 0) {
    return { query: cleanQuery, category, count: 0, results: [] };
  }

  // Sort candidate chunk IDs by fused RRF score descending
  const sortedEntries = Array.from(scores.values()).sort((a, b) => b.rrfScore - a.rrfScore);
  const topEntries = sortedEntries.slice(0, limit);
  const topIds = topEntries.map((e) => e.chunkId);

  // 4. Hydrate chunk records
  const chunksMap = getChunksByIds(category, topIds);

  const results = [];
  for (const entry of topEntries) {
    const chunk = chunksMap.get(entry.chunkId);
    if (!chunk) continue;

    results.push({
      id: chunk.id,
      docPath: chunk.doc_path,
      title: chunk.title,
      section: chunk.section,
      lineStart: chunk.line_start,
      lineEnd: chunk.line_end,
      content: chunk.content,
      score: Number(entry.rrfScore.toFixed(5)),
      denseScore: Number(entry.denseScore.toFixed(4)),
      denseRank: entry.denseRank,
      bm25Rank: entry.bm25Rank
    });
  }

  return {
    query: cleanQuery,
    category,
    count: results.length,
    results
  };
}

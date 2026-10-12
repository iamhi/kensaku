import { embedText } from '../pipeline/embeddings.js';
import { knnSearch, bm25Search, getChunksByIds } from '../db/sqliteManager.js';
import { rerankCandidates, DEFAULT_RERANKER_MODEL } from '../pipeline/reranker.js';

export { rerankCandidates, DEFAULT_RERANKER_MODEL };

const RRF_K = 60; // Standard Reciprocal Rank Fusion constant
const MIN_SIMILARITY_THRESHOLD = 0.25; // Noise floor for dense embeddings

/**
 * Merges adjacent or overlapping chunks from the same document with proximity boost.
 * @param {Array<object>} hits
 * @param {number} [proximityLines=5]
 * @returns {Array<object>}
 */
export function mergeNearbyChunks(hits, proximityLines = 5) {
  if (!hits || hits.length <= 1) return hits || [];

  // Group by workspace:docPath
  const byDoc = new Map();
  for (const hit of hits) {
    const key = `${hit.workspace || 'default'}::${hit.docPath}`;
    const list = byDoc.get(key) || [];
    list.push({ ...hit });
    byDoc.set(key, list);
  }

  const mergedResults = [];

  for (const docHits of byDoc.values()) {
    // Sort chronologically by start line
    docHits.sort((a, b) => a.lineStart - b.lineStart);
    let current = docHits[0];

    for (let i = 1; i < docHits.length; i++) {
      const next = docHits[i];
      if (next.lineStart <= current.lineEnd + proximityLines) {
        // Contiguous / overlapping chunks: extend line boundaries
        current.lineEnd = Math.max(current.lineEnd, next.lineEnd);
        current.content = `${current.content}\n\n---\n\n${next.content}`;
        if (current.section !== next.section && !current.section.includes(next.section)) {
          current.section = `${current.section} & ${next.section}`;
        }
        // Controlled proximity score accumulation
        current.score = Number((Math.max(current.score, next.score) + 0.15 * Math.min(current.score, next.score)).toFixed(5));
        current.denseScore = Math.max(current.denseScore, next.denseScore);
      } else {
        mergedResults.push(current);
        current = next;
      }
    }
    mergedResults.push(current);
  }

  return mergedResults.sort((a, b) => b.score - a.score);
}

/**
 * Executes a hybrid (Dense Vector KNN + Sparse FTS5 BM25) search with Reciprocal Rank Fusion (RRF).
 * @param {string} category - Workspace category name.
 * @param {string} query - Search query text.
 * @param {object} [options]
 * @param {number} [options.limit=10] - Number of final chunks to return.
 * @param {number} [options.candidateLimit=30] - Candidate pool size for fusion.
 * @param {boolean} [options.mergeProximity=true] - Whether to merge nearby chunks.
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

  // 3. Reciprocal Rank Fusion (RRF) with similarity calibration
  const scores = new Map(); // chunkId -> { rrfScore, denseScore, bm25Score, denseRank, bm25Rank }

  // Process Dense KNN ranks with similarity gating
  knnResults.forEach((item, index) => {
    const rank = index + 1;
    // Cosine distance to similarity: 1 - distance
    const similarity = Math.max(0, 1 - (item.distance || 0));

    // Exclude noise hits below similarity threshold unless they also match lexical BM25
    if (similarity < MIN_SIMILARITY_THRESHOLD) {
      return;
    }

    // Similarity-scaled RRF to solve cross-collection rank disparity
    const rrfContrib = similarity / (RRF_K + rank);

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
  const topEntries = sortedEntries.slice(0, candidateLimit);
  const topIds = topEntries.map((e) => e.chunkId);

  // 4. Hydrate chunk records
  const chunksMap = getChunksByIds(category, topIds);

  const candidates = [];
  for (const entry of topEntries) {
    const chunk = chunksMap.get(entry.chunkId);
    if (!chunk) continue;

    candidates.push({
      id: chunk.id,
      workspace: chunk.category || category,
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

  // 5. Optional Proximity Merge
  const mergedCandidates = options.mergeProximity !== false
    ? mergeNearbyChunks(candidates, 5)
    : candidates;

  // 6. Stage-2 Cross-Encoder Reranking
  const shouldRerank = (options.rerank === true || options.rerankModel || options.customReranker) && options.noRerank !== true;
  const finalResults = shouldRerank
    ? await rerankCandidates(cleanQuery, mergedCandidates, { ...options, limit })
    : mergedCandidates.slice(0, limit);

  return {
    query: cleanQuery,
    category,
    count: finalResults.length,
    results: finalResults
  };
}

/**
 * Searches across multiple workspaces in parallel, merging and ranking results globally.
 * @param {string[]} categories - Array of workspace category names.
 * @param {string} query - Search query text.
 * @param {object} [options]
 * @returns {Promise<{ query: string, category: string, count: number, results: Array<object> }>}
 */
export async function searchMultipleWorkspaces(categories, query, options = {}) {
  const limit = options.limit || 10;
  if (!categories || categories.length === 0) {
    return { query, category: 'all', count: 0, results: [] };
  }

  // Execute first-stage retrieval without sub-limiting
  const searchPromises = categories.map((cat) =>
    hybridSearch(cat, query, { ...options, rerank: false, limit: limit * 2 })
      .catch((err) => {
        console.warn(`[Kensaku] Search failed for workspace "${cat}":`, err.message);
        return { results: [] };
      })
  );

  const searchResponses = await Promise.all(searchPromises);
  const allHits = [];

  for (const resp of searchResponses) {
    allHits.push(...(resp.results || []));
  }

  // Sort merged results by fused score descending
  allHits.sort((a, b) => b.score - a.score);

  // Apply proximity merge across documents
  const mergedHits = options.mergeProximity !== false
    ? mergeNearbyChunks(allHits, 5)
    : allHits;

  // Stage-2 Cross-Encoder Reranking across all combined workspaces
  const shouldRerank = (options.rerank === true || options.rerankModel || options.customReranker) && options.noRerank !== true;
  const finalResults = shouldRerank
    ? await rerankCandidates(query, mergedHits, { ...options, limit })
    : mergedHits.slice(0, limit);

  return {
    query,
    category: categories.length === 1 ? categories[0] : 'all',
    count: finalResults.length,
    results: finalResults
  };
}

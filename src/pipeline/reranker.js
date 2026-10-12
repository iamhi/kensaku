import path from 'node:path';
import os from 'node:os';
import { pipeline, env } from '@xenova/transformers';

// Configure model cache in home directory
env.cacheDir = path.join(os.homedir(), '.cache', 'xenova');

export const DEFAULT_RERANKER_MODEL = 'Xenova/bge-reranker-base';

let rerankerPipelinePromise = null;
let currentLoadedModel = null;

/**
 * Returns a cached text-classification pipeline for cross-encoder reranking.
 * @param {string} [modelName=DEFAULT_RERANKER_MODEL]
 * @returns {Promise<Function>}
 */
export async function getRerankerPipeline(modelName = DEFAULT_RERANKER_MODEL) {
  if (!rerankerPipelinePromise || currentLoadedModel !== modelName) {
    currentLoadedModel = modelName;
    rerankerPipelinePromise = pipeline('text-classification', modelName, {
      quantized: true
    }).catch((err) => {
      rerankerPipelinePromise = null;
      currentLoadedModel = null;
      throw err;
    });
  }
  return rerankerPipelinePromise;
}

/**
 * Rescores candidate chunks against a query using a Cross-Encoder model.
 * Evaluates deep cross-attention between the query and each chunk's content.
 * 
 * @param {string} query - The search query.
 * @param {Array<object>} candidates - Chunks returned from Stage-1 hybrid search.
 * @param {object} [options]
 * @param {boolean} [options.rerank=true] - Set to false or use options.noRerank to bypass reranking.
 * @param {boolean} [options.noRerank=false] - When true, skips cross-encoder evaluation.
 * @param {string} [options.model=DEFAULT_RERANKER_MODEL] - Model identifier.
 * @param {number} [options.batchSize=8] - Max pairs per inference batch to prevent memory pressure.
 * @param {number} [options.limit] - Limit of top results to return.
 * @param {number} [options.maxPassageChars=2000] - Max characters per passage passed to cross-encoder.
 * @param {Function} [options.customReranker] - Optional custom rerank function (query, candidate) => score.
 * @returns {Promise<Array<object>>} - Rescored and sorted candidate chunks.
 */
export async function rerankCandidates(query, candidates, options = {}) {
  if (!candidates || candidates.length === 0) {
    return [];
  }

  // Check explicit bypass flags
  if (options.rerank === false || options.noRerank === true) {
    const limit = options.limit || candidates.length;
    return candidates.slice(0, limit);
  }

  const cleanQuery = (query || '').trim();
  if (!cleanQuery) {
    return candidates;
  }

  const batchSize = options.batchSize || 8;
  const maxChars = options.maxPassageChars || 2000;
  const modelName = options.model || DEFAULT_RERANKER_MODEL;

  // Custom / mock reranker support (for testing or remote TEI endpoints)
  if (typeof options.customReranker === 'function') {
    try {
      const scored = await Promise.all(
        candidates.map(async (c) => {
          const score = await options.customReranker(cleanQuery, c);
          return {
            ...c,
            rerankScore: Number(Number(score).toFixed(5)),
            originalScore: c.score,
            score: Number(Number(score).toFixed(5))
          };
        })
      );
      scored.sort((a, b) => b.rerankScore - a.rerankScore);
      return options.limit ? scored.slice(0, options.limit) : scored;
    } catch (err) {
      console.warn('[Kensaku] Custom reranker error, falling back to RRF ranking:', err.message);
      return candidates;
    }
  }

  // Cross-Encoder via @xenova/transformers
  try {
    const classifier = await getRerankerPipeline(modelName);
    const scoredCandidates = [];

    for (let i = 0; i < candidates.length; i += batchSize) {
      const batchChunks = candidates.slice(i, i + batchSize);
      const batchPromises = batchChunks.map(async (chunk) => {
        const passage = (chunk.content || '').slice(0, maxChars);
        try {
          const out = await classifier(cleanQuery, { text_pair: passage });
          // Score represents semantic relevance / classification probability
          const score = Array.isArray(out) ? (out[0]?.score ?? 0) : (out?.score ?? 0);
          return {
            ...chunk,
            rerankScore: Number(Number(score).toFixed(5)),
            originalScore: chunk.score,
            score: Number(Number(score).toFixed(5))
          };
        } catch {
          // If individual pair evaluation errors, keep original score
          return {
            ...chunk,
            rerankScore: null,
            originalScore: chunk.score
          };
        }
      });

      const batchResults = await Promise.all(batchPromises);
      scoredCandidates.push(...batchResults);
    }

    scoredCandidates.sort((a, b) => {
      if (a.rerankScore != null && b.rerankScore != null) {
        return b.rerankScore - a.rerankScore;
      }
      return b.score - a.score;
    });

    const limit = options.limit || scoredCandidates.length;
    return scoredCandidates.slice(0, limit);
  } catch (err) {
    // Graceful fallback: If cross-encoder model cannot be loaded (e.g. offline first-run),
    // warn and return original RRF candidates without failing the search
    if (process.env.DEBUG || options.verbose) {
      console.warn(`[Kensaku] Cross-encoder reranker unavailable (${err.message}). Using RRF ranking.`);
    }
    const limit = options.limit || candidates.length;
    return candidates.slice(0, limit);
  }
}

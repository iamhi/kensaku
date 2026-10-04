import os from 'node:os';
import path from 'node:path';
import { pipeline, env } from '@xenova/transformers';

// Configure model cache in home directory
env.cacheDir = path.join(os.homedir(), '.cache', 'xenova');

let extractorPromise = null;
const DEFAULT_MODEL = 'Xenova/all-MiniLM-L6-v2';

/**
 * Returns a singleton feature-extraction pipeline.
 * @param {string} [modelName]
 * @returns {Promise<Function>}
 */
export async function getEmbeddingPipeline(modelName = DEFAULT_MODEL) {
  if (!extractorPromise) {
    extractorPromise = pipeline('feature-extraction', modelName, {
      quantized: true
    });
  }
  return extractorPromise;
}

/**
 * Generates a normalized Float32Array embedding vector for a single text.
 * @param {string} text
 * @param {object} [options]
 * @returns {Promise<Float32Array>}
 */
export async function embedText(text, options = {}) {
  const extractor = await getEmbeddingPipeline(options.model);
  const clean = (text || '').trim();
  if (!clean) {
    return new Float32Array(384);
  }
  const output = await extractor(clean, {
    pooling: 'mean',
    normalize: true
  });
  return new Float32Array(output.data);
}

/**
 * Generates embedding vectors for an array of texts with bounded concurrency.
 * @param {string[]} texts
 * @param {object} [options]
 * @param {number} [options.concurrency=8]
 * @returns {Promise<Float32Array[]>}
 */
export async function embedBatch(texts, options = {}) {
  const extractor = await getEmbeddingPipeline(options.model);
  const concurrency = options.concurrency || 8;
  const results = new Array(texts.length);

  for (let i = 0; i < texts.length; i += concurrency) {
    const chunk = texts.slice(i, i + concurrency);
    const promises = chunk.map(async (text, idx) => {
      const clean = (text || '').trim();
      if (!clean) {
        return new Float32Array(384);
      }
      const output = await extractor(clean, {
        pooling: 'mean',
        normalize: true
      });
      return new Float32Array(output.data);
    });
    const batchOutputs = await Promise.all(promises);
    for (let j = 0; j < batchOutputs.length; j++) {
      results[i + j] = batchOutputs[j];
    }
  }

  return results;
}

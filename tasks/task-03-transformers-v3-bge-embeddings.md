# Task 03: Upgrade to Transformers.js v3 & BGE Asymmetric Embeddings

## Objective
Upgrade the transformers runtime from legacy `@xenova/transformers` (v2) to `@huggingface/transformers` (v3) in [`package.json`](file:///Users/darkoorovchanec/Documents/workplace/playground/oni/kensaku/package.json), and transition from `all-MiniLM-L6-v2` to `Xenova/bge-small-en-v1.5` with asymmetric query instruction prefixes in [`src/pipeline/embeddings.js`](file:///Users/darkoorovchanec/Documents/workplace/playground/oni/kensaku/src/pipeline/embeddings.js).

---

## Background & Problem
1. **Legacy Dependency**: `@xenova/transformers` is deprecated in favor of official Hugging Face package `@huggingface/transformers` v3, which supports modern quantized dtypes (`q8`, `q4`), WebGPU, and ONNX Runtime performance improvements.
2. **Retrieval Quality**: `all-MiniLM-L6-v2` is an older symmetric model that underperforms on technical documentation retrieval compared to `bge-small-en-v1.5`.
3. **Asymmetric Gap**: The BGE model family requires an asymmetric instruction prefix on queries (`"Represent this sentence for searching relevant passages: "`) to map short search queries to comprehensive explanatory passages.

---

## Technical Specifications

### 1. Update Package Dependencies
In [`package.json`](file:///Users/darkoorovchanec/Documents/workplace/playground/oni/kensaku/package.json):
- Replace `"@xenova/transformers": "2.17.2"` with `"@huggingface/transformers": "^3.3.3"`.

### 2. Implement BGE Embedder with Asymmetric Query Prefix
In [`src/pipeline/embeddings.js`](file:///Users/darkoorovchanec/Documents/workplace/playground/oni/kensaku/src/pipeline/embeddings.js):
```javascript
import { pipeline, env } from '@huggingface/transformers';

const DEFAULT_MODEL = 'Xenova/bge-small-en-v1.5';
const QUERY_INSTRUCTION = 'Represent this sentence for searching relevant passages: ';

let extractorPromise = null;

export async function getEmbeddingPipeline(modelName = DEFAULT_MODEL) {
  if (!extractorPromise) {
    extractorPromise = pipeline('feature-extraction', modelName, {
      dtype: 'q8'
    });
  }
  return extractorPromise;
}

export async function embedText(text, options = {}) {
  const extractor = await getEmbeddingPipeline(options.model);
  const clean = (text || '').trim();
  if (!clean) return new Float32Array(384);

  const input = options.isQuery ? `${QUERY_INSTRUCTION}${clean}` : clean;
  const output = await extractor(input, { pooling: 'mean', normalize: true });
  return new Float32Array(output.data);
}
```

### 3. Update Callers
Update [`src/index.js`](file:///Users/darkoorovchanec/Documents/workplace/playground/oni/kensaku/src/index.js#L237-L238) and [`src/services/searchEngine.js`](file:///Users/darkoorovchanec/Documents/workplace/playground/oni/kensaku/src/services/searchEngine.js#L75) to pass `{ isQuery: true }` when generating query vectors and `{ isQuery: false }` when indexing document chunks.

---

## Verification & Acceptance Criteria
- [ ] Embedding dimension remains 384, retaining compatibility with `sqlite-vec vec0(embedding float[384])`.
- [ ] Model weights download cleanly into the local cache directory (`~/.cache/kensaku/` or default Hugging Face cache).
- [ ] `kensaku test` reports PASS for the in-process ONNX embedder under `@huggingface/transformers` v3.

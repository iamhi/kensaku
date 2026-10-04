# Task 04: Stage-2 Batched Cross-Encoder Reranking

## Objective
Implement a second-stage Cross-Encoder reranking pipeline in [`src/services/searchEngine.js`](file:///Users/darkoorovchanec/Documents/workplace/playground/oni/kensaku/src/services/searchEngine.js) using `Xenova/bge-reranker-base` (Q8) to rescore candidate chunks from the initial Dense + BM25 Reciprocal Rank Fusion stage.

---

## Background & Problem
1. **Bi-Encoder Limitations**: Bi-encoders compute independent embeddings for queries and documents, comparing them via cosine distance. This misses fine-grained token-level cross-attention, causing false-positive keyword matches to outrank genuinely informative answers.
2. **Current Status in Kensaku**: Currently, [`hybridSearch()`](file:///Users/darkoorovchanec/Documents/workplace/playground/oni/kensaku/src/services/searchEngine.js#L65-L174) returns RRF candidates directly without deep cross-encoder evaluation.
3. **CLI Performance Constraint**: Reranking must be batched to avoid ONNX thread contention, and users should have a `--no-rerank` flag for sub-20ms ultra-low latency searches.

---

## Technical Specifications

### 1. Cross-Encoder Model Loader
In [`src/services/searchEngine.js`](file:///Users/darkoorovchanec/Documents/workplace/playground/oni/kensaku/src/services/searchEngine.js):
```javascript
import { pipeline } from '@huggingface/transformers';

let rerankerPromise = null;

export async function getRerankerPipeline(modelName = 'Xenova/bge-reranker-base') {
  if (!rerankerPromise) {
    rerankerPromise = pipeline('text-classification', modelName, {
      dtype: 'q8'
    });
  }
  return rerankerPromise;
}
```

### 2. Batched Rescoring Logic
After Stage-1 RRF produces the top 15 candidate chunks:
1. Format query-chunk pairs:
   ```javascript
   const pairs = topCandidates.map((c) => ({
     text: query,
     text_pair: c.content
   }));
   ```
2. Run inference in bounded batches (e.g., batches of 8) rather than unbatched `Promise.all`:
   ```javascript
   const reranker = await getRerankerPipeline();
   const scoredResults = [];
   
   for (let i = 0; i < pairs.length; i += 8) {
     const batch = pairs.slice(i, i + 8);
     const batchOutputs = await reranker(batch);
     batchOutputs.forEach((output, idx) => {
       scoredResults.push({
         ...topCandidates[i + idx],
         rerankScore: output.score
       });
     });
   }
   
   return scoredResults
     .sort((a, b) => b.rerankScore - a.rerankScore)
     .slice(0, limit);
   ```

### 3. Add `--no-rerank` Flag
Allow skipping Stage 2 when execution speed is prioritized:
```bash
kensaku search "PostgreSQL WAL" --no-rerank
```

---

## Verification & Acceptance Criteria
- [ ] Top-15 candidates from RRF are rescored with `bge-reranker-base`.
- [ ] Precision of top 3 chunks improves on queries where keywords overlap with unrelated sections.
- [ ] Adding `--no-rerank` skips the reranker and returns RRF fused candidates directly.
- [ ] Latency for 15 pairs remains within ~50–90 ms on CPU hardware.

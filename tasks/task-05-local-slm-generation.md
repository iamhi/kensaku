# Task 05: In-Process Local ONNX SLM with Terminal Streaming

## Objective
Introduce an in-process Small Language Model (SLM) generator powered by ONNX Runtime and `@huggingface/transformers` in [`src/services/llmClient.js`](file:///Users/darkoorovchanec/Documents/workplace/playground/oni/kensaku/src/services/llmClient.js), enabling 100% offline, zero-API-key Question Answering with real-time terminal token streaming.

---

## Background & Problem
1. **Dependency on External Services**: Currently, `kensaku ask` fails if the user does not have an active API key (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`) or an active local `ollama` daemon.
2. **True Air-Gapped Privacy**: For private, offline, or confidential codebases, having a self-contained in-process model guarantees zero data leaves the machine without requiring external setup.
3. **CLI Streaming Requirement**: Generation must stream tokens interactively to the terminal as they are produced rather than blocking until completion.

---

## Technical Specifications

### 1. In-Process ONNX Model Configuration
Support `onnx-community/Qwen2.5-1.5B-Instruct` (Q4, ~1.1 GB) as default local model, with fallback support for `onnx-community/Qwen2.5-0.5B-Instruct` (Q4, ~350 MB) for resource-constrained environments:
```javascript
import { pipeline, TextStreamer } from '@huggingface/transformers';

let localGeneratorPromise = null;

async function getLocalGenerator(modelName = 'onnx-community/Qwen2.5-1.5B-Instruct') {
  if (!localGeneratorPromise) {
    localGeneratorPromise = pipeline('text-generation', modelName, {
      dtype: 'q4'
    });
  }
  return localGeneratorPromise;
}
```

### 2. Streaming Generator Integration
In [`src/services/llmClient.js`](file:///Users/darkoorovchanec/Documents/workplace/playground/oni/kensaku/src/services/llmClient.js), implement `streamLocalONNX()`:
```javascript
async function streamLocalONNX({ config, systemPrompt, userPrompt, onToken }) {
  const generator = await getLocalGenerator(config.model);
  let fullText = '';

  const streamer = new TextStreamer(generator.tokenizer, {
    skip_prompt: true,
    skip_special_tokens: true,
    callback_function: (token) => {
      fullText += token;
      if (onToken) onToken(token);
    }
  });

  const messages = [
    { role: 'system', content: systemPrompt },
    { role: 'user', content: userPrompt }
  ];

  await generator(messages, {
    max_new_tokens: 350,
    temperature: 0.1,
    do_sample: false,
    streamer
  });

  return { text: fullText, provider: 'local', model: config.model };
}
```

### 3. Provider Resolution Hierarchy
Update [`resolveLLMConfig()`](file:///Users/darkoorovchanec/Documents/workplace/playground/oni/kensaku/src/services/llmClient.js#L11-L47):
1. Explicit `--provider` option (e.g. `--provider=anthropic`, `--provider=ollama`, `--provider=local`).
2. Cloud API keys if present (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`).
3. If local Ollama is active (`localhost:11434`), use Ollama.
4. **Default fallback**: If no keys are found and Ollama is offline, default to `provider: 'local'` (in-process ONNX Qwen2.5).

---

## Verification & Acceptance Criteria
- [ ] Running `kensaku ask "What is SQLite WAL?"` with no API keys and Ollama stopped successfully generates an answer using local ONNX.
- [ ] Tokens stream in real time to the terminal via `process.stdout.write`.
- [ ] Explicitly setting `--provider=anthropic` or `--provider=ollama` still routes to external providers.
- [ ] Citations format accurately as `[Source 1]` or line references.

/**
 * Minimalist, zero-dependency LLM client supporting local Ollama, Anthropic Claude,
 * Google Gemini, and OpenAI-compatible endpoints with native streaming.
 */

/**
 * Detects the active or requested LLM provider based on environment variables and options.
 * @param {object} [options]
 * @returns {{ provider: 'ollama' | 'openai' | 'anthropic' | 'gemini', model: string, apiKey?: string, baseUrl?: string }}
 */
export function resolveLLMConfig(options = {}) {
  const provider = (options.provider || process.env.KENSAKU_LLM_PROVIDER || '').toLowerCase();

  if (provider === 'anthropic' || (!provider && process.env.ANTHROPIC_API_KEY)) {
    return {
      provider: 'anthropic',
      model: options.model || process.env.KENSAKU_LLM_MODEL || 'claude-3-5-haiku-20241022',
      apiKey: process.env.ANTHROPIC_API_KEY,
      baseUrl: options.baseUrl || process.env.ANTHROPIC_BASE_URL || 'https://api.anthropic.com/v1'
    };
  }

  if (provider === 'gemini' || (!provider && process.env.GEMINI_API_KEY)) {
    return {
      provider: 'gemini',
      model: options.model || process.env.KENSAKU_LLM_MODEL || 'gemini-2.0-flash',
      apiKey: process.env.GEMINI_API_KEY,
      baseUrl: options.baseUrl || 'https://generativelanguage.googleapis.com/v1beta'
    };
  }

  if (provider === 'openai' || (!provider && process.env.OPENAI_API_KEY)) {
    return {
      provider: 'openai',
      model: options.model || process.env.KENSAKU_LLM_MODEL || 'gpt-4o-mini',
      apiKey: process.env.OPENAI_API_KEY,
      baseUrl: options.baseUrl || process.env.OPENAI_BASE_URL || 'https://api.openai.com/v1'
    };
  }

  // Default to local Ollama
  return {
    provider: 'ollama',
    model: options.model || process.env.KENSAKU_LLM_MODEL || process.env.OLLAMA_MODEL || 'llama3.2',
    baseUrl: options.baseUrl || process.env.OLLAMA_BASE_URL || process.env.OLLAMA_HOST || 'http://localhost:11434'
  };
}

/**
 * Streams completion tokens from an LLM.
 * @param {object} params
 * @param {string} params.systemPrompt
 * @param {string} params.userPrompt
 * @param {object} [params.options]
 * @param {(token: string) => void} [params.onToken] - Callback for each streamed token
 * @returns {Promise<{ text: string, provider: string, model: string }>}
 */
export async function streamCompletion({ systemPrompt, userPrompt, options = {}, onToken = null }) {
  const config = resolveLLMConfig(options);

  switch (config.provider) {
    case 'ollama':
      return streamOllama({ config, systemPrompt, userPrompt, onToken });
    case 'anthropic':
      return streamAnthropic({ config, systemPrompt, userPrompt, onToken });
    case 'gemini':
      return streamGemini({ config, systemPrompt, userPrompt, onToken });
    case 'openai':
    default:
      return streamOpenAI({ config, systemPrompt, userPrompt, onToken });
  }
}

/**
 * Checks if the configured LLM provider is reachable.
 * @param {object} [options]
 * @returns {Promise<{ ok: boolean, provider: string, model: string, error?: string }>}
 */
export async function checkLLMHealth(options = {}) {
  const config = resolveLLMConfig(options);
  try {
    if (config.provider === 'ollama') {
      const res = await fetch(`${config.baseUrl}/api/tags`, { signal: AbortSignal.timeout(3000) });
      if (!res.ok) throw new Error(`Ollama returned status ${res.status}`);
      const data = await res.json();
      const models = (data.models || []).map((m) => m.name);
      return { ok: true, provider: 'ollama', model: config.model, availableModels: models };
    }

    if (!config.apiKey) {
      return { ok: false, provider: config.provider, model: config.model, error: `Missing API key for ${config.provider}` };
    }

    return { ok: true, provider: config.provider, model: config.model };
  } catch (err) {
    return { ok: false, provider: config.provider, model: config.model, error: err.message };
  }
}

// ---------------------------------------------------------------------------
// Provider Implementations (Native Fetch + Streaming SSE / NDJSON)
// ---------------------------------------------------------------------------

async function streamOllama({ config, systemPrompt, userPrompt, onToken }) {
  let response;
  try {
    response = await fetch(`${config.baseUrl}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: config.model,
        messages: [
          { role: 'system', content: systemPrompt },
          { role: 'user', content: userPrompt }
        ],
        stream: true
      })
    });
  } catch (err) {
    throw new Error(
      `Failed to connect to Ollama at ${config.baseUrl}. Is Ollama running? Run 'ollama run ${config.model}' or set ANTHROPIC_API_KEY/OPENAI_API_KEY/GEMINI_API_KEY. (${err.message})`
    );
  }

  if (!response.ok) {
    const errText = await response.text();
    throw new Error(`Ollama error (${response.status}): ${errText}`);
  }

  let fullText = '';
  const reader = response.body.getReader();
  const decoder = new TextDecoder('utf8');
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) continue;
      try {
        const parsed = JSON.parse(trimmed);
        const token = parsed.message?.content || '';
        if (token) {
          fullText += token;
          if (onToken) onToken(token);
        }
      } catch {}
    }
  }

  return { text: fullText, provider: 'ollama', model: config.model };
}

async function streamOpenAI({ config, systemPrompt, userPrompt, onToken }) {
  if (!config.apiKey) {
    throw new Error('OPENAI_API_KEY environment variable is required for OpenAI provider.');
  }

  const response = await fetch(`${config.baseUrl}/chat/completions`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${config.apiKey}`
    },
    body: JSON.stringify({
      model: config.model,
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: userPrompt }
      ],
      stream: true
    })
  });

  if (!response.ok) {
    const errText = await response.text();
    throw new Error(`OpenAI API error (${response.status}): ${errText}`);
  }

  let fullText = '';
  const reader = response.body.getReader();
  const decoder = new TextDecoder('utf8');
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed || !trimmed.startsWith('data: ')) continue;
      const dataStr = trimmed.slice(6).trim();
      if (dataStr === '[DONE]') continue;

      try {
        const parsed = JSON.parse(dataStr);
        const token = parsed.choices?.[0]?.delta?.content || '';
        if (token) {
          fullText += token;
          if (onToken) onToken(token);
        }
      } catch {}
    }
  }

  return { text: fullText, provider: 'openai', model: config.model };
}

async function streamAnthropic({ config, systemPrompt, userPrompt, onToken }) {
  if (!config.apiKey) {
    throw new Error('ANTHROPIC_API_KEY environment variable is required for Anthropic provider.');
  }

  const response = await fetch(`${config.baseUrl}/messages`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'x-api-key': config.apiKey,
      'anthropic-version': '2023-06-01'
    },
    body: JSON.stringify({
      model: config.model,
      max_tokens: 4096,
      system: systemPrompt,
      messages: [{ role: 'user', content: userPrompt }],
      stream: true
    })
  });

  if (!response.ok) {
    const errText = await response.text();
    throw new Error(`Anthropic API error (${response.status}): ${errText}`);
  }

  let fullText = '';
  const reader = response.body.getReader();
  const decoder = new TextDecoder('utf8');
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed || !trimmed.startsWith('data: ')) continue;
      const dataStr = trimmed.slice(6).trim();

      try {
        const parsed = JSON.parse(dataStr);
        if (parsed.type === 'content_block_delta' && parsed.delta?.text) {
          const token = parsed.delta.text;
          fullText += token;
          if (onToken) onToken(token);
        }
      } catch {}
    }
  }

  return { text: fullText, provider: 'anthropic', model: config.model };
}

async function streamGemini({ config, systemPrompt, userPrompt, onToken }) {
  if (!config.apiKey) {
    throw new Error('GEMINI_API_KEY environment variable is required for Gemini provider.');
  }

  const url = `${config.baseUrl}/models/${config.model}:streamGenerateContent?alt=sse&key=${config.apiKey}`;
  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      contents: [{ role: 'user', parts: [{ text: userPrompt }] }],
      systemInstruction: { parts: [{ text: systemPrompt }] }
    })
  });

  if (!response.ok) {
    const errText = await response.text();
    throw new Error(`Gemini API error (${response.status}): ${errText}`);
  }

  let fullText = '';
  const reader = response.body.getReader();
  const decoder = new TextDecoder('utf8');
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed || !trimmed.startsWith('data: ')) continue;
      const dataStr = trimmed.slice(6).trim();

      try {
        const parsed = JSON.parse(dataStr);
        const candidate = parsed.candidates?.[0];
        const parts = candidate?.content?.parts || [];
        for (const p of parts) {
          if (p.text) {
            fullText += p.text;
            if (onToken) onToken(p.text);
          }
        }
      } catch {}
    }
  }

  return { text: fullText, provider: 'gemini', model: config.model };
}

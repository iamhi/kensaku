import { spawn, spawnSync } from 'node:child_process';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import crypto from 'node:crypto';

/**
 * Minimalist, zero-dependency LLM client supporting local Yoakai CLI harness,
 * Ollama, Anthropic Claude, Google Gemini, and OpenAI-compatible endpoints with native streaming.
 */

/**
 * Checks whether the yoakai CLI binary is installed and executable in PATH.
 * @returns {boolean}
 */
export function isYoakaiAvailable() {
  try {
    const isWin = process.platform === 'win32';
    const checkCmd = isWin ? 'where' : 'which';
    const res = spawnSync(checkCmd, ['yoakai'], { stdio: 'pipe' });
    return res.status === 0;
  } catch {
    return false;
  }
}

/**
 * Detects the active or requested LLM provider based on environment variables and options.
 * Hierarchy:
 *   1. Explicit CLI / Option: options.provider (e.g. 'yoakai', 'anthropic', etc.)
 *   2. Environment Priority: KENSAKU_LLM_PROVIDER or KENSAKU_PREFER_YOAKAI=true
 *   3. Configured Cloud API keys (ANTHROPIC_API_KEY, GEMINI_API_KEY, OPENAI_API_KEY)
 *   4. Default Fallback: Yoakai if installed in PATH
 *   5. Final Fallback: Local Ollama
 * @param {object} [options]
 * @returns {{ provider: 'yoakai' | 'ollama' | 'openai' | 'anthropic' | 'gemini', model?: string, harness?: string, effort?: string, apiKey?: string, baseUrl?: string }}
 */
export function resolveLLMConfig(options = {}) {
  const provider = (options.provider || process.env.KENSAKU_LLM_PROVIDER || '').toLowerCase();
  const preferYoakai = process.env.KENSAKU_PREFER_YOAKAI === 'true';

  // 1. Explicitly requested Yoakai or prioritized via env
  if (provider === 'yoakai' || preferYoakai) {
    return {
      provider: 'yoakai',
      harness: options.harness || process.env.KENSAKU_YOAKAI_HARNESS,
      model: options.model || process.env.KENSAKU_LLM_MODEL,
      effort: options.effort || process.env.KENSAKU_YOAKAI_EFFORT
    };
  }

  // 2. Explicitly requested or configured cloud providers
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

  // 3. Explicitly requested Ollama
  if (provider === 'ollama') {
    return {
      provider: 'ollama',
      model: options.model || process.env.KENSAKU_LLM_MODEL || process.env.OLLAMA_MODEL || 'llama3.2',
      baseUrl: options.baseUrl || process.env.OLLAMA_BASE_URL || process.env.OLLAMA_HOST || 'http://localhost:11434'
    };
  }

  // 4. Default Fallback: If nothing is configured, check if Yoakai is available
  if (isYoakaiAvailable()) {
    return {
      provider: 'yoakai',
      harness: options.harness || process.env.KENSAKU_YOAKAI_HARNESS,
      model: options.model || process.env.KENSAKU_LLM_MODEL,
      effort: options.effort || process.env.KENSAKU_YOAKAI_EFFORT
    };
  }

  // 5. Final fallback to local Ollama
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
 * @returns {Promise<{ text: string, provider: string, model?: string, harness?: string }>}
 */
export async function streamCompletion({ systemPrompt, userPrompt, options = {}, onToken = null }) {
  const config = resolveLLMConfig(options);

  switch (config.provider) {
    case 'yoakai':
      return streamYoakai({ config, systemPrompt, userPrompt, onToken });
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
 * @returns {Promise<{ ok: boolean, provider: string, model?: string, harness?: string, effort?: string, availableModels?: string[], error?: string }>}
 */
export async function checkLLMHealth(options = {}) {
  const config = resolveLLMConfig(options);
  try {
    if (config.provider === 'yoakai') {
      if (!isYoakaiAvailable()) {
        return { ok: false, provider: 'yoakai', error: 'yoakai command not found in PATH' };
      }
      try {
        const isWin = process.platform === 'win32';
        const res = spawnSync('yoakai', ['config'], {
          encoding: 'utf8',
          shell: isWin,
          timeout: 4000
        });
        let details = {};
        if (res.stdout) {
          try {
            details = JSON.parse(res.stdout);
          } catch {}
        }
        return {
          ok: true,
          provider: 'yoakai',
          harness: config.harness || details.harness || 'agy',
          model: config.model || details.model || 'default',
          effort: config.effort || details.effort
        };
      } catch (err) {
        return { ok: true, provider: 'yoakai', harness: config.harness || 'default', model: config.model || 'default' };
      }
    }

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

/**
 * Streams answer generation via Yoakai CLI.
 * Yoakai routes prompts through agy, claude, or copilot harnesses.
 * @param {object} params
 * @param {object} params.config
 * @param {string} params.systemPrompt
 * @param {string} params.userPrompt
 * @param {(token: string) => void} [params.onToken]
 * @returns {Promise<{ text: string, provider: 'yoakai', model?: string, harness?: string }>}
 */
async function streamYoakai({ config, systemPrompt, userPrompt, onToken }) {
  if (!isYoakaiAvailable()) {
    throw new Error('Yoakai CLI is not installed or not found in PATH. Install Yoakai or specify another provider.');
  }

  // Construct combined prompt for Yoakai
  const fullPrompt = `${systemPrompt ? `# Instructions\n${systemPrompt}\n\n` : ''}${userPrompt}`;
  const tempDir = os.tmpdir();
  const tempFilePath = path.join(tempDir, `kensaku-prompt-${crypto.randomUUID()}.md`);

  await fs.writeFile(tempFilePath, fullPrompt, 'utf8');

  const isWin = process.platform === 'win32';
  const yoakaiArgs = [tempFilePath];

  // Only pass parameters if explicitly configured - let Yoakai use its own defaults otherwise
  if (config.harness) {
    yoakaiArgs.push(`--harness=${config.harness}`);
  }
  if (config.model) {
    yoakaiArgs.push(`--model=${config.model}`);
  }
  if (config.effort) {
    yoakaiArgs.push(`--effort=${config.effort}`);
  }

  return new Promise((resolve, reject) => {
    let fullText = '';
    let stderrText = '';

    const child = spawn('yoakai', yoakaiArgs, {
      stdio: ['inherit', 'pipe', 'pipe'],
      shell: isWin
    });

    child.stdout.on('data', (chunk) => {
      const str = chunk.toString();
      fullText += str;
      if (onToken) {
        onToken(str);
      }
    });

    child.stderr.on('data', (chunk) => {
      stderrText += chunk.toString();
    });

    child.on('error', async (err) => {
      try {
        await fs.unlink(tempFilePath);
      } catch {}
      reject(new Error(`Failed to execute Yoakai: ${err.message}`));
    });

    child.on('close', async (code) => {
      try {
        await fs.unlink(tempFilePath);
      } catch {}

      if (code !== 0) {
        const errorMsg = stderrText.trim() || `Yoakai process exited with code ${code}`;
        reject(new Error(`Yoakai execution failed: ${errorMsg}`));
        return;
      }

      resolve({
        text: fullText,
        provider: 'yoakai',
        harness: config.harness,
        model: config.model
      });
    });
  });
}


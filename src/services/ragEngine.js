import { hybridSearch, searchMultipleWorkspaces } from './searchEngine.js';
import { getWorkspacesRegistry } from '../db/sqliteManager.js';
import { streamCompletion } from './llmClient.js';
import { renderCitations, auditCitations } from './citationRenderer.js';

/**
 * Reorders candidate chunks to mitigate the "Lost in the Middle" attention phenomenon.
 * Places the highest relevance chunks at the beginning and end of the context window.
 * @param {Array<object>} chunks - Sorted by score descending.
 * @returns {Array<object>}
 */
export function reorderLostInTheMiddle(chunks) {
  if (!chunks || chunks.length <= 2) return chunks ? [...chunks] : [];

  const reordered = new Array(chunks.length);
  let left = 0;
  let right = chunks.length - 1;

  for (let i = 0; i < chunks.length; i++) {
    if (i % 2 === 0) {
      reordered[left++] = chunks[i];
    } else {
      reordered[right--] = chunks[i];
    }
  }

  return reordered;
}

/**
 * Formats retrieved chunks into unambiguous XML document envelopes with metadata.
 * @param {Array<object>} chunks
 * @returns {string}
 */
export function formatContextEnvelopes(chunks) {
  if (!chunks || chunks.length === 0) {
    return '<context>\n  <!-- No matching documents retrieved -->\n</context>';
  }

  const envelopes = chunks.map((chunk, index) => {
    const docId = index + 1;
    const workspace = chunk.workspace || 'default';
    const cleanContent = (chunk.content || '').trim();

    return `  <document id="${docId}" workspace="${workspace}" path="${chunk.docPath}" lines="${chunk.lineStart}-${chunk.lineEnd}" title="${chunk.title || ''}" section="${chunk.section || ''}" score="${chunk.score}">
${cleanContent}
  </document>`;
  });

  return `<context>\n${envelopes.join('\n\n')}\n</context>`;
}

/**
 * Constructs a strict, grounded system prompt enforcing citations and preventing hallucinations.
 * @returns {string}
 */
export function buildGroundedSystemPrompt() {
  return `You are Kensaku (検索), an expert, precise, and faithful technical retrieval assistant.

Your task is to answer the user's question STRICTLY and EXCLUSIVELY based on the provided <context> documents.

CRITICAL INSTRUCTIONS:
1. Grounding: Every statement, explanation, or code snippet you provide MUST be supported by facts explicitly stated in the <context>.
2. Citations: You MUST cite your sources using bracketed document IDs (e.g. [1], [2]) or path line references (e.g. [docs/api.md#L10-L25]) for every factual claim.
3. Missing Information: If the provided documents do not contain enough facts to answer the question with certainty, you MUST state clearly: "Based on the provided workspace documents, I do not have sufficient information to answer this question." Do NOT guess or hallucinate.
4. Tone & Style: Be concise, clear, and direct. Provide code blocks when relevant with language syntax highlighting.`;
}

/**
 * Executes the complete RAG lifecycle: retrieval, context curation, envelope structuring,
 * and grounded LLM generation with streaming tokens.
 * @param {object} params
 * @param {string} params.query - User question/prompt.
 * @param {string} [params.category] - Target workspace category ('all' or specific name).
 * @param {number} [params.limit=5] - Number of top chunks to include in context.
 * @param {object} [params.llmOptions] - LLM provider/model configuration.
 * @param {(token: string) => void} [params.onToken] - Streaming token callback.
 * @returns {Promise<{ answer: string, query: string, category: string, sources: Array<object>, durationMs: number }>}
 */
export async function executeRAG({
  query,
  category = 'all',
  limit = 5,
  rerank = undefined,
  noRerank = undefined,
  llmOptions = {},
  onToken = null
}) {
  const startTime = Date.now();
  const cleanQuery = (query || '').trim();

  if (!cleanQuery) {
    throw new Error('Query cannot be empty for RAG execution.');
  }

  // 1. Multi-Workspace or Single-Workspace Hybrid Retrieval
  let searchResult;
  const normCategory = (category || 'all').trim().toLowerCase();
  const searchOpts = { limit, rerank, noRerank, ...llmOptions };

  if (normCategory === 'all') {
    const registry = getWorkspacesRegistry();
    const registeredCategories = Object.keys(registry);
    if (registeredCategories.length === 0) {
      const msg = 'No workspaces are registered in Kensaku. Add a workspace first using: kensaku add workspace <name> <path>';
      if (onToken) onToken(msg);
      return {
        answer: msg,
        query: cleanQuery,
        category: 'all',
        sources: [],
        durationMs: Date.now() - startTime
      };
    }
    searchResult = await searchMultipleWorkspaces(registeredCategories, cleanQuery, searchOpts);
  } else {
    searchResult = await hybridSearch(normCategory, cleanQuery, searchOpts);
  }

  const rawChunks = searchResult.results || [];

  if (rawChunks.length === 0) {
    const fallbackAnswer = `No relevant documents were found in workspace "${normCategory}" for your query: "${cleanQuery}".`;
    if (onToken) onToken(fallbackAnswer);
    return {
      answer: fallbackAnswer,
      query: cleanQuery,
      category: normCategory,
      sources: [],
      durationMs: Date.now() - startTime
    };
  }

  // 2. Lost-in-the-Middle Context Reordering
  const curatedChunks = reorderLostInTheMiddle(rawChunks);

  // 3. Document Envelopes Formatting
  const contextXml = formatContextEnvelopes(curatedChunks);

  // 4. Construct Prompt
  const systemPrompt = buildGroundedSystemPrompt();
  const userPrompt = `${contextXml}\n\nUser Question: ${cleanQuery}\n\nPlease provide a grounded answer with citations:`;

  // 5. Stream LLM Generation
  const completion = await streamCompletion({
    systemPrompt,
    userPrompt,
    options: llmOptions,
    onToken
  });

  const durationMs = Date.now() - startTime;

  const formattedSources = curatedChunks.map((c, i) => ({
    id: i + 1,
    workspace: c.workspace,
    docPath: c.docPath,
    lineStart: c.lineStart,
    lineEnd: c.lineEnd,
    title: c.title,
    section: c.section,
    score: c.score
  }));

  // Linkify citations and strip hallucinated references
  const linkifiedAnswer = renderCitations(completion.text, formattedSources, {
    format: llmOptions?.citationFormat || 'markdown',
    workspaceRoot: llmOptions?.workspaceRoot || null,
    stripHallucinations: llmOptions?.stripHallucinations !== false
  });

  const citationAudit = auditCitations(completion.text, formattedSources);

  return {
    answer: linkifiedAnswer,
    rawAnswer: completion.text,
    query: cleanQuery,
    category: normCategory,
    provider: completion.provider,
    harness: completion.harness,
    model: completion.model,
    sources: formattedSources,
    citations: citationAudit,
    durationMs
  };
}

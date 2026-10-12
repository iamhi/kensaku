import path from 'node:path';

/**
 * Checks whether an inner bracket string consists strictly of comma-separated numeric IDs.
 * Example: "1", "1, 2", "1, 3, 5"
 * @param {string} str
 * @returns {boolean}
 */
const NUMERIC_CITATION_REGEX = /^\s*\d+(?:\s*,\s*\d+)*\s*$/;

/**
 * Finds the matching closing bracket in a string, respecting backslash escapes.
 * @param {string} line
 * @param {number} start - Index of the opening '['
 * @returns {number} Index of matching ']', or -1 if unclosed
 */
function findClosingBracket(line, start) {
  let depth = 0;
  for (let i = start; i < line.length; i++) {
    if (line[i] === '\\') {
      i++;
      continue;
    }
    if (line[i] === '[') depth++;
    if (line[i] === ']' && --depth === 0) return i;
  }
  return -1;
}

/**
 * Finds the matching closing parenthesis in a string.
 * @param {string} line
 * @param {number} start - Index of the opening '('
 * @returns {number}
 */
function findClosingParen(line, start) {
  let depth = 0;
  for (let i = start; i < line.length; i++) {
    if (line[i] === '\\') {
      i++;
      continue;
    }
    if (line[i] === '(') depth++;
    if (line[i] === ')' && --depth === 0) return i;
  }
  return -1;
}

/**
 * Generates a clickable URI for a source chunk.
 * @param {object} source
 * @param {string} [workspaceRoot]
 * @returns {string}
 */
export function buildSourceUri(source, workspaceRoot) {
  if (!source) return '#';
  const rawPath = source.docPath || source.path || 'unknown';
  let targetPath = rawPath;

  if (workspaceRoot) {
    targetPath = path.isAbsolute(rawPath) ? rawPath : path.resolve(workspaceRoot, rawPath);
  }

  // Normalize windows backslashes to forward slashes for URI scheme
  const normalizedPath = targetPath.replace(/\\/g, '/');
  const lineHash = source.lineStart
    ? source.lineEnd && source.lineEnd !== source.lineStart
      ? `#L${source.lineStart}-L${source.lineEnd}`
      : `#L${source.lineStart}`
    : '';

  if (normalizedPath.startsWith('/') || /^[a-zA-Z]:\//.test(normalizedPath)) {
    return `file://${normalizedPath}${lineHash}`;
  }

  return `${normalizedPath}${lineHash}`;
}

/**
 * Code-aware markdown citation processor.
 * 
 * Accurately parses markdown responses from LLMs, protects fenced code blocks,
 * indentation, and inline ticks from corrupting code literals (e.g. array[0]),
 * validates citation numbers against retrieved sources, strips hallucinated/phantom
 * citations, and converts valid citations into clickable hyperlinks.
 * 
 * @param {string} message - Raw markdown answer text from LLM.
 * @param {Array<object>} sources - Array of retrieved chunk objects.
 * @param {object} [options]
 * @param {'markdown'|'terminal'} [options.format='markdown'] - Output format.
 * @param {string} [options.workspaceRoot] - Absolute workspace root for file:// links.
 * @param {boolean} [options.stripHallucinations=true] - Remove out-of-range citations (e.g. [9]).
 * @returns {string} Processed text with verified, clickable citation links.
 */
export function renderCitations(message, sources = [], options = {}) {
  if (typeof message !== 'string' || !message.trim()) {
    return message || '';
  }

  const refs = Array.isArray(sources) ? sources : [];
  const format = options.format || 'markdown';
  const workspaceRoot = options.workspaceRoot || null;
  const stripHallucinations = options.stripHallucinations !== false;

  let fence = null;
  let inline = null;

  function formatCitationLink(index, source) {
    const uri = buildSourceUri(source, workspaceRoot);
    const titleAttr = source.title ? ` "${source.title.replace(/"/g, "'")}"` : '';

    if (format === 'terminal') {
      // OSC 8 terminal hyperlink or clean terminal citation
      const isTTY = typeof process !== 'undefined' && process.stdout?.isTTY;
      if (isTTY) {
        return `\u001b]8;;${uri}\u001b\\[${index}]\u001b]8;;\u001b\\`;
      }
      return `[${index}](${uri})`;
    }

    // Standard markdown link
    return `[${index}](${uri}${titleAttr})`;
  }

  function transformCitationToken(token, inner) {
    if (!NUMERIC_CITATION_REGEX.test(inner)) {
      return token;
    }

    const parts = inner.split(',').map((p) => p.trim());
    const validLinks = [];

    for (const part of parts) {
      const idx = Number(part);
      if (!Number.isSafeInteger(idx) || idx < 1) {
        continue;
      }

      // Check bounds against retrieved source array (1-indexed)
      const source = refs[idx - 1];
      if (source) {
        validLinks.push(formatCitationLink(idx, source));
      } else if (!stripHallucinations) {
        // Keep unlinked marker if hallucination stripping is disabled
        validLinks.push(`[${idx}]`);
      }
    }

    if (validLinks.length === 0) {
      // Stripped hallucination or no sources
      return '';
    }

    return validLinks.join(' ');
  }

  function processProseLine(line) {
    let output = '';

    for (let i = 0; i < line.length; ) {
      // 1. Handle active inline code span
      if (inline) {
        const closeIdx = line.indexOf(inline, i);
        if (closeIdx === -1) {
          output += line.slice(i);
          break;
        }
        output += line.slice(i, closeIdx + inline.length);
        i = closeIdx + inline.length;
        inline = null;
        continue;
      }

      // 2. Check for start of inline code tick (`` or `)
      if (line[i] === '`') {
        let end = i + 1;
        while (end < line.length && line[end] === '`') end++;
        const tickMarker = line.slice(i, end);
        output += tickMarker;
        inline = tickMarker;
        i = end;
        continue;
      }

      // 3. Check for opening bracket of citation candidate
      if (line[i] !== '[') {
        output += line[i];
        i++;
        continue;
      }

      const closeIdx = findClosingBracket(line, i);
      if (closeIdx === -1) {
        output += line.slice(i);
        break;
      }

      const token = line.slice(i, closeIdx + 1);
      const inner = token.slice(1, -1);
      const nextChar = line[closeIdx + 1];

      // If already followed by '(' or '[', it's already a markdown link or footnote definition
      if (nextChar === '(') {
        const parenClose = findClosingParen(line, closeIdx + 1);
        if (parenClose !== -1) {
          // Existing markdown link like [label](url): leave intact
          output += line.slice(i, parenClose + 1);
          i = parenClose + 1;
          continue;
        }
      }

      // If preceded by '!', it's an image: ![alt]
      if (i > 0 && line[i - 1] === '!') {
        output += token;
        i = closeIdx + 1;
        continue;
      }

      // Transform citation token
      const replaced = transformCitationToken(token, inner);
      output += replaced;
      i = closeIdx + 1;
    }

    return output;
  }

  const lines = message.split('\n');
  const processedLines = lines.map((line) => {
    // Check for fenced code block start/end: ``` or ~~~
    const fenceMatch = /^(?: {0,3})(`{3,}|~{3,})/.exec(line);
    if (fence) {
      if (fenceMatch && line.trim().startsWith(fence.marker[0]) && line.trim().length >= fence.marker.length) {
        fence = null; // Close fence
      }
      return line;
    }

    if (fenceMatch) {
      fence = { marker: fenceMatch[1] };
      return line;
    }

    // 4-space or tab indented code block
    if (/^(?: {4}|\t)/.test(line)) {
      return line;
    }

    return processProseLine(line);
  });

  return processedLines.join('\n');
}

/**
 * Extracts and inspects all citation references present in a message.
 * @param {string} message
 * @param {Array<object>} sources
 * @returns {{ citations: number[], validCitations: number[], hallucinatedCitations: number[], citedSources: object[] }}
 */
export function auditCitations(message, sources = []) {
  if (!message || typeof message !== 'string') {
    return { citations: [], validCitations: [], hallucinatedCitations: [], citedSources: [] };
  }

  const allCitations = new Set();
  const validCitations = new Set();
  const hallucinatedCitations = new Set();

  const regex = /\[(\d+(?:\s*,\s*\d+)*)\]/g;
  let match;

  while ((match = regex.exec(message)) !== null) {
    const parts = match[1].split(',').map((p) => parseInt(p.trim(), 10));
    for (const num of parts) {
      if (Number.isSafeInteger(num) && num > 0) {
        allCitations.add(num);
        if (num <= sources.length) {
          validCitations.add(num);
        } else {
          hallucinatedCitations.add(num);
        }
      }
    }
  }

  const citedSources = Array.from(validCitations)
    .sort((a, b) => a - b)
    .map((idx) => sources[idx - 1])
    .filter(Boolean);

  return {
    citations: Array.from(allCitations).sort((a, b) => a - b),
    validCitations: Array.from(validCitations).sort((a, b) => a - b),
    hallucinatedCitations: Array.from(hallucinatedCitations).sort((a, b) => a - b),
    citedSources
  };
}

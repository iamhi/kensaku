import crypto from 'node:crypto';

/**
 * Computes a deterministic SHA-256 hash for document content.
 * @param {string} content
 * @returns {string}
 */
export function hashContent(content) {
  return crypto.createHash('sha256').update(content || '', 'utf8').digest('hex');
}

/**
 * Splits a markdown document into section-aware semantic chunks with line ranges.
 * @param {string} content - Raw markdown text.
 * @param {string} filePath - Relative file path.
 * @param {object} [options]
 * @param {number} [options.maxChunkChars=1200] - Max character length per chunk.
 * @param {number} [options.overlapChars=200] - Overlap length for oversized sections.
 * @returns {Array<{ chunkIndex: number, docPath: string, title: string, section: string, lineStart: number, lineEnd: number, content: string }>}
 */
export function chunkMarkdown(content, filePath, options = {}) {
  const maxChunkChars = options.maxChunkChars || 1200;
  const overlapChars = options.overlapChars || 200;

  if (!content || !content.trim()) {
    return [];
  }

  const lines = content.split('\n');
  const sections = [];
  let currentTitle = filePath;
  let currentSection = 'Introduction';
  let sectionLines = [];
  let sectionStartLine = 1;

  // Detect top-level title if present early in the document
  for (let i = 0; i < Math.min(lines.length, 10); i++) {
    const trimmed = lines[i].trim();
    if (trimmed.startsWith('# ')) {
      currentTitle = trimmed.replace(/^#\s+/, '').trim();
      break;
    }
  }

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const lineNum = i + 1;
    const headingMatch = line.match(/^(#{1,4})\s+(.+)$/);

    if (headingMatch && sectionLines.length > 0) {
      // Flush previous section
      const sectionText = sectionLines.join('\n').trim();
      if (sectionText.length > 0) {
        sections.push({
          title: currentTitle,
          section: currentSection,
          lineStart: sectionStartLine,
          lineEnd: lineNum - 1,
          text: sectionText
        });
      }
      currentSection = headingMatch[2].trim();
      if (headingMatch[1] === '#') {
        currentTitle = currentSection;
      }
      sectionLines = [line];
      sectionStartLine = lineNum;
    } else {
      if (headingMatch) {
        currentSection = headingMatch[2].trim();
        if (headingMatch[1] === '#') {
          currentTitle = currentSection;
        }
        sectionStartLine = lineNum;
      }
      sectionLines.push(line);
    }
  }

  // Flush remaining lines
  if (sectionLines.length > 0) {
    const sectionText = sectionLines.join('\n').trim();
    if (sectionText.length > 0) {
      sections.push({
        title: currentTitle,
        section: currentSection,
        lineStart: sectionStartLine,
        lineEnd: lines.length,
        text: sectionText
      });
    }
  }

  const finalChunks = [];
  let chunkIndex = 0;

  for (const sec of sections) {
    if (sec.text.length <= maxChunkChars) {
      finalChunks.push({
        chunkIndex: chunkIndex++,
        docPath: filePath,
        title: sec.title,
        section: sec.section,
        lineStart: sec.lineStart,
        lineEnd: sec.lineEnd,
        content: sec.text
      });
    } else {
      // Split large section with sliding window along paragraph/line boundaries
      const secLines = sec.text.split('\n');
      let windowLines = [];
      let windowChars = 0;
      let windowStartLine = sec.lineStart;

      for (let j = 0; j < secLines.length; j++) {
        const line = secLines[j];
        windowLines.push(line);
        windowChars += line.length + 1;

        if (windowChars >= maxChunkChars || j === secLines.length - 1) {
          const windowEndLine = sec.lineStart + j;
          const chunkContent = windowLines.join('\n').trim();

          if (chunkContent.length > 0) {
            finalChunks.push({
              chunkIndex: chunkIndex++,
              docPath: filePath,
              title: sec.title,
              section: sec.section,
              lineStart: windowStartLine,
              lineEnd: windowEndLine,
              content: chunkContent
            });
          }

          // Compute overlap
          const preservedLines = [];
          let preservedChars = 0;
          for (let k = windowLines.length - 1; k >= 0; k--) {
            if (preservedChars + windowLines[k].length <= overlapChars) {
              preservedLines.unshift(windowLines[k]);
              preservedChars += windowLines[k].length + 1;
            } else {
              break;
            }
          }

          windowLines = preservedLines;
          windowChars = preservedChars;
          windowStartLine = Math.max(sec.lineStart, windowEndLine - preservedLines.length + 1);
        }
      }
    }
  }

  return finalChunks;
}

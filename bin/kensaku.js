#!/usr/bin/env node

import path from 'node:path';
import {
  addWorkspace,
  removeWorkspace,
  listWorkspaces,
  indexWorkspace,
  searchWorkspace,
  getWorkspaceStatus,
  closeAllDatabases
} from '../src/index.js';

const args = process.argv.slice(2);
const command = args[0];

function printHelp() {
  console.log(`
🔍 KENSAKU (検索) — Semantic Search & Embedding Engine

Usage:
  kensaku search "<query>" [workspace <name>] [--limit=<n>]
  kensaku add workspace <name> <path>
  kensaku remove workspace <name>
  kensaku index <name> [--force]
  kensaku status [name]
  kensaku list
  kensaku help

Examples:
  kensaku add workspace kawara /home/iamhi/yoakai_workspace
  kensaku index kawara
  kensaku search "frontier ai research" workspace kawara
  kensaku status kawara
`);
}

async function main() {
  if (!command || command === 'help' || command === '--help' || command === '-h') {
    printHelp();
    return;
  }

  try {
    if (command === 'add' && args[1] === 'workspace') {
      const name = args[2];
      const dirPath = args[3];
      if (!name || !dirPath) {
        console.error('Error: specify workspace name and path. Example: kensaku add workspace dev ./docs');
        process.exit(1);
      }
      const absPath = path.resolve(process.cwd(), dirPath);
      const reg = addWorkspace(name, absPath);
      console.log(`[Kensaku] Registered workspace "${name}" -> ${reg.path}`);
      console.log(`[Kensaku] Starting initial indexing for "${name}"...`);
      const res = await indexWorkspace(name);
      console.log(`[Kensaku] Indexed ${res.scanned} files (${res.totalChunks} chunks) in ${res.durationMs}ms`);
      return;
    }

    if (command === 'remove' && args[1] === 'workspace') {
      const name = args[2];
      if (!name) {
        console.error('Error: specify workspace name to remove');
        process.exit(1);
      }
      removeWorkspace(name);
      console.log(`[Kensaku] Removed workspace "${name}"`);
      return;
    }

    if (command === 'index') {
      const name = args[1];
      if (!name) {
        console.error('Error: specify workspace name to index');
        process.exit(1);
      }
      const force = args.includes('--force');
      console.log(`[Kensaku] Indexing workspace "${name}"${force ? ' (forced full re-index)' : ''}...`);
      const res = await indexWorkspace(name, { force });
      console.log(`[Kensaku] Finished in ${res.durationMs}ms:`);
      console.log(`  - Scanned: ${res.scanned}`);
      console.log(`  - Updated / New: ${res.updated}`);
      console.log(`  - Unchanged: ${res.unchanged}`);
      console.log(`  - Deleted: ${res.deleted}`);
      console.log(`  - Total Chunks: ${res.totalChunks}`);
      return;
    }

    if (command === 'search') {
      const query = args[1];
      if (!query) {
        console.error('Error: query cannot be empty. Example: kensaku search "deep learning" workspace dev');
        process.exit(1);
      }

      // Check for workspace flag: workspace <name> or --workspace=<name>
      let workspaceName = 'default';
      let limit = 5;

      for (let i = 2; i < args.length; i++) {
        if (args[i] === 'workspace' && args[i + 1]) {
          workspaceName = args[i + 1];
          i++;
        } else if (args[i].startsWith('--limit=')) {
          limit = parseInt(args[i].split('=')[1], 10) || 5;
        } else if (args[i].startsWith('--workspace=')) {
          workspaceName = args[i].split('=')[1];
        }
      }

      const res = await searchWorkspace(workspaceName, query, { limit });
      console.log(`\n🔍 Found ${res.count} results for "${query}" in workspace "${workspaceName}":\n`);
      res.results.forEach((item, idx) => {
        console.log(`[${idx + 1}] ${item.title} (${item.docPath}#L${item.lineStart}-L${item.lineEnd})`);
        console.log(`    Section: ${item.section} | Score: ${item.score} (Dense: ${item.denseScore})`);
        const preview = item.content.split('\n').slice(0, 3).join(' ').trim();
        console.log(`    Snippet: ${preview.length > 120 ? preview.slice(0, 120) + '...' : preview}\n`);
      });
      return;
    }

    if (command === 'list') {
      const list = listWorkspaces();
      if (list.length === 0) {
        console.log('[Kensaku] No workspaces registered yet. Add one with: kensaku add workspace <name> <path>');
        return;
      }
      console.log('\nRegistered Workspaces:');
      list.forEach((w) => {
        console.log(`  • ${w.category.padEnd(16)} -> ${w.path} (${w.totalDocs} docs, ${w.totalChunks} chunks)`);
      });
      console.log();
      return;
    }

    if (command === 'status') {
      const name = args[1];
      if (name) {
        const status = getWorkspaceStatus(name);
        console.log(JSON.stringify(status, null, 2));
      } else {
        const list = listWorkspaces();
        console.log(JSON.stringify(list, null, 2));
      }
      return;
    }

    console.error(`Unknown command: ${command}`);
    printHelp();
    process.exit(1);
  } finally {
    closeAllDatabases();
  }
}

main().catch((err) => {
  console.error('[Kensaku] Fatal error:', err);
  closeAllDatabases();
  process.exit(1);
});

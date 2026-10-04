#!/usr/bin/env node

import path from 'node:path';
import {
  addWorkspace,
  removeWorkspace,
  listWorkspaces,
  indexWorkspace,
  searchWorkspace,
  askWorkspace,
  getWorkspaceStatus,
  checkWorkspaceStaleness,
  closeAllDatabases,
  checkLLMHealth
} from '../src/index.js';
import { embedText } from '../src/pipeline/embeddings.js';
import Database from 'better-sqlite3';
import * as sqliteVec from 'sqlite-vec';

const args = process.argv.slice(2);
const command = args[0];

function printHelp() {
  console.log(`
🔍 KENSAKU (検索) — Semantic Search & Full RAG Engine

Usage:
  kensaku search "<query>" [workspace <name>] [--limit=<n>] [--no-sync]
  kensaku ask "<question>" [workspace <name>] [--limit=<n>] [--model=<m>] [--provider=<p>]
  kensaku add workspace <name> <path>
  kensaku remove workspace <name>
  kensaku index <name> [--force]
  kensaku status [name]
  kensaku list
  kensaku test
  kensaku help

Commands:
  search     Semantic hybrid search (across all workspaces by default, or single workspace)
  ask        Execute full RAG: retrieve, lost-in-the-middle reorder, stream grounded answer
  add        Register a directory as a named workspace and run initial indexing
  remove     Delete a workspace and its SQLite vector database
  index      Index or re-index a workspace with incremental SHA-256 change detection
  status     Inspect workspace statistics, document counts, and composite hash
  list       Show all registered workspaces
  test       Run self-diagnostic integrity check on SQLite, vectors, embeddings, and LLM

Examples:
  kensaku search "What are good standards for Java?"
  kensaku search "Connection pooling" workspace backend --limit=3
  kensaku ask "How is the storage layer implemented?"
  kensaku ask "Explain the recipe steps" workspace cooking
  kensaku add workspace cooking ./recipes
  kensaku test
`);
}

async function runSelfTest() {
  console.log('\n🧪 Running Kensaku Self-Diagnostics...\n');
  let allPassed = true;

  // 1. SQLite & sqlite-vec Extension
  process.stdout.write('  1. SQLite & sqlite-vec vector extension: ');
  try {
    const db = new Database(':memory:');
    sqliteVec.load(db);
    db.exec(`
      CREATE VIRTUAL TABLE test_vec USING vec0(embedding float[4]);
      INSERT INTO test_vec(rowid, embedding) VALUES (1, '[1.0, 0.0, 0.0, 0.0]');
    `);
    const hit = db.prepare('SELECT rowid, distance FROM test_vec WHERE embedding MATCH ? AND k = 1')
      .get(new Float32Array([1.0, 0.0, 0.0, 0.0]));
    db.close();

    if (hit && hit.rowid === 1) {
      console.log('✅ PASS');
    } else {
      throw new Error('Vector match did not return expected row');
    }
  } catch (err) {
    console.log(`❌ FAIL (${err.message})`);
    allPassed = false;
  }

  // 2. FTS5 BM25 Full-Text Search
  process.stdout.write('  2. SQLite FTS5 BM25 lexical engine: ');
  try {
    const db = new Database(':memory:');
    db.exec(`
      CREATE VIRTUAL TABLE test_fts USING fts5(content);
      INSERT INTO test_fts(content) VALUES ('Kensaku full text search engine');
    `);
    const ftsHit = db.prepare('SELECT rowid, bm25(test_fts) as rank FROM test_fts WHERE test_fts MATCH ?')
      .get('Kensaku');
    db.close();

    if (ftsHit) {
      console.log('✅ PASS');
    } else {
      throw new Error('FTS5 search did not return expected row');
    }
  } catch (err) {
    console.log(`❌ FAIL (${err.message})`);
    allPassed = false;
  }

  // 3. Local ONNX Transformer Embeddings
  process.stdout.write('  3. In-process ONNX embeddings (@xenova/transformers): ');
  try {
    const vec = await embedText('Kensaku diagnostic test');
    if (vec instanceof Float32Array && vec.length === 384) {
      console.log(`✅ PASS (384-dim normalized vector generated)`);
    } else {
      throw new Error(`Unexpected vector dimensions: ${vec?.length}`);
    }
  } catch (err) {
    console.log(`❌ FAIL (${err.message})`);
    allPassed = false;
  }

  // 4. Workspace Registry
  process.stdout.write('  4. Workspace registry and database directory: ');
  try {
    const workspaces = listWorkspaces();
    console.log(`✅ PASS (${workspaces.length} workspace(s) registered)`);
  } catch (err) {
    console.log(`❌ FAIL (${err.message})`);
    allPassed = false;
  }

  // 5. LLM Provider Reachability
  process.stdout.write('  5. LLM provider configuration: ');
  try {
    const llmHealth = await checkLLMHealth();
    if (llmHealth.ok) {
      console.log(`✅ PASS (${llmHealth.provider}: ${llmHealth.model})`);
    } else {
      console.log(`⚠️  NOT READY (${llmHealth.provider}: ${llmHealth.error || 'unreachable'}. Set ANTHROPIC_API_KEY/OPENAI_API_KEY/GEMINI_API_KEY or start Ollama)`);
    }
  } catch (err) {
    console.log(`⚠️  WARNING (${err.message})`);
  }

  console.log(allPassed ? '\n✨ All core systems are operational!\n' : '\n⚠️  Some diagnostics failed. Please inspect the logs above.\n');
  return allPassed ? 0 : 1;
}

async function main() {
  if (!command || command === 'help' || command === '--help' || command === '-h') {
    printHelp();
    return;
  }

  try {
    if (command === 'test') {
      const exitCode = await runSelfTest();
      process.exit(exitCode);
    }

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
      console.log(`[Kensaku] Indexed ${res.scanned} files (${res.totalChunks} chunks, composite hash: ${res.compositeHash.slice(0, 12)}...) in ${res.durationMs}ms`);
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
      console.log(`  - Composite Hash: ${res.compositeHash}`);
      return;
    }

    if (command === 'search') {
      const query = args[1];
      if (!query) {
        console.error('Error: query cannot be empty. Example: kensaku search "deep learning"');
        process.exit(1);
      }

      let workspaceName = 'all'; // Default to all workspaces per idea.md
      let limit = 5;
      let autoSync = true;

      for (let i = 2; i < args.length; i++) {
        if (args[i] === 'workspace' && args[i + 1]) {
          workspaceName = args[i + 1];
          i++;
        } else if (args[i].startsWith('--limit=')) {
          limit = parseInt(args[i].split('=')[1], 10) || 5;
        } else if (args[i].startsWith('--workspace=')) {
          workspaceName = args[i].split('=')[1];
        } else if (args[i] === '--no-sync') {
          autoSync = false;
        }
      }

      const res = await searchWorkspace(workspaceName, query, { limit, autoSync });
      const scopeLabel = workspaceName === 'all' ? 'all workspaces' : `workspace "${workspaceName}"`;
      console.log(`\n🔍 Found ${res.count} results for "${query}" across ${scopeLabel}:\n`);

      if (res.results.length === 0) {
        console.log('  No matching documents found.');
        return;
      }

      res.results.forEach((item, idx) => {
        const wsTag = item.workspace ? `[${item.workspace}] ` : '';
        console.log(`[${idx + 1}] ${wsTag}${item.title} (${item.docPath}#L${item.lineStart}-L${item.lineEnd})`);
        console.log(`    Section: ${item.section} | Score: ${item.score} (Dense: ${item.denseScore})`);
        const preview = item.content.split('\n').slice(0, 3).join(' ').trim();
        console.log(`    Snippet: ${preview.length > 140 ? preview.slice(0, 140) + '...' : preview}\n`);
      });
      return;
    }

    if (command === 'ask') {
      const prompt = args[1];
      if (!prompt) {
        console.error('Error: prompt cannot be empty. Example: kensaku ask "How does SQLite WAL mode work?"');
        process.exit(1);
      }

      let workspaceName = 'all';
      let limit = 5;
      let model = undefined;
      let provider = undefined;

      for (let i = 2; i < args.length; i++) {
        if (args[i] === 'workspace' && args[i + 1]) {
          workspaceName = args[i + 1];
          i++;
        } else if (args[i].startsWith('--limit=')) {
          limit = parseInt(args[i].split('=')[1], 10) || 5;
        } else if (args[i].startsWith('--workspace=')) {
          workspaceName = args[i].split('=')[1];
        } else if (args[i].startsWith('--model=')) {
          model = args[i].split('=')[1];
        } else if (args[i].startsWith('--provider=')) {
          provider = args[i].split('=')[1];
        }
      }

      const scopeLabel = workspaceName === 'all' ? 'all workspaces' : `workspace "${workspaceName}"`;
      console.log(`\n🤖 Asking Kensaku (searching ${scopeLabel})...\n`);

      let tokenCount = 0;
      const onToken = (token) => {
        tokenCount++;
        process.stdout.write(token);
      };

      try {
        const ragRes = await askWorkspace(workspaceName, prompt, {
          limit,
          autoSync: true,
          llmOptions: { model, provider },
          onToken
        });

        if (ragRes.sources && ragRes.sources.length > 0) {
          console.log('\n\n' + '─'.repeat(60));
          console.log(`📚 Grounded Citations & Sources (${ragRes.durationMs}ms via ${ragRes.provider || 'local'}/${ragRes.model || 'model'}):`);
          ragRes.sources.forEach((src) => {
            const wsTag = src.workspace ? `[${src.workspace}] ` : '';
            console.log(`  [${src.id}] ${wsTag}${src.docPath}#L${src.lineStart}-L${src.lineEnd} — "${src.section}" (score: ${src.score})`);
          });
          console.log();
        } else {
          console.log('\n');
        }
      } catch (err) {
        console.error(`\n❌ RAG generation failed: ${err.message}`);
        console.log('\n💡 Tip: You can run semantic search without an LLM:');
        console.log(`   kensaku search "${prompt}" workspace ${workspaceName}\n`);
        process.exit(1);
      }
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
        const hashDisplay = w.compositeHash ? ` (hash: ${w.compositeHash.slice(0, 8)})` : '';
        console.log(`  • ${w.category.padEnd(16)} -> ${w.path} (${w.totalDocs} docs, ${w.totalChunks} chunks)${hashDisplay}`);
      });
      console.log();
      return;
    }

    if (command === 'status') {
      const name = args[1];
      if (name) {
        const status = getWorkspaceStatus(name);
        const staleness = await checkWorkspaceStaleness(name);
        console.log(JSON.stringify({ ...status, staleness }, null, 2));
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

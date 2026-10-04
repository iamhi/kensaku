---
name: kensaku
description: Local semantic search and full RAG engine over indexed workspaces using SQLite, sqlite-vec, FTS5 BM25, and ONNX embeddings.
---

# Kensaku (検索) — Semantic Search & Full RAG Skill

Kensaku is a zero-external-service semantic search and Retrieval-Augmented Generation (RAG) tool. It indexes markdown and code documentation in named workspaces into local SQLite databases (`better-sqlite3` + `sqlite-vec` + FTS5) using local ONNX embeddings (`Xenova/all-MiniLM-L6-v2`).

## CLI Usage

### 1. Full RAG Execution (Ask a question and get grounded answer with citations)
```bash
# Query all workspaces
kensaku ask "How is the storage layer implemented?"

# Query a specific workspace
kensaku ask "What is the authentic carbonara recipe?" workspace cooking

# Specify an LLM provider and model (supports Ollama, Anthropic, Gemini, OpenAI)
kensaku ask "Explain the architecture" workspace dev --provider=anthropic --model=claude-3-5-haiku-20241022
```

### 2. Semantic Search (Retrieve relevant chunks without LLM generation)
```bash
# Search across all registered workspaces
kensaku search "semantic search and sqlite-vec"

# Search a single workspace with custom limit
kensaku search "carbonara pecorino" workspace cooking --limit=3
```

### 3. Workspace Management
```bash
# Register a directory as a workspace (indexes automatically)
kensaku add workspace <name> <path_to_directory>

# List all registered workspaces with document & chunk counts
kensaku list

# Check workspace statistics, staleness, and composite SHA-256 hash
kensaku status <name>

# Force re-index of a workspace
kensaku index <name> --force

# Remove a workspace and its SQLite database
kensaku remove workspace <name>
```

### 4. Self-Diagnostics
```bash
# Test SQLite, sqlite-vec, FTS5, ONNX transformer embeddings, and LLM configuration
kensaku test
```

## Programmatic Node.js API

```javascript
import {
  searchWorkspace,
  askWorkspace,
  addWorkspace,
  listWorkspaces,
  indexWorkspace,
  checkWorkspaceStaleness
} from 'kensaku';

// 1. Search all workspaces
const results = await searchWorkspace('all', 'vector database');

// 2. Full RAG execution with streaming tokens
const rag = await askWorkspace('dev', 'How do transactions work in SQLite?', {
  onToken: (token) => process.stdout.write(token)
});
console.log('\nAnswer:', rag.answer);
console.log('Citations:', rag.sources);
```

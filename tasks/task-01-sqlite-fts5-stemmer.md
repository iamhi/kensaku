# Task 01: SQLite FTS5 Porter Stemmer & Synchronization Triggers

## Objective
Upgrade the SQLite FTS5 lexical full-text search table in [`src/db/sqliteManager.js`](file:///Users/darkoorovchanec/Documents/workplace/playground/oni/kensaku/src/db/sqliteManager.js) to use the English Porter stemmer and unicode diacritics removal, and add the missing `AFTER UPDATE` trigger to ensure relational-to-FTS synchronization.

---

## Background & Problem
1. Currently, [`getDatabase()`](file:///Users/darkoorovchanec/Documents/workplace/playground/oni/kensaku/src/db/sqliteManager.js#L132-L138) initializes `chunks_fts` with default tokenization:
   ```sql
   CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
     content, title, section, content='chunks', content_rowid='id'
   );
   ```
   This performs raw prefix/exact matching without stemming. Searches for `"canceling"` miss `"cancellation"`, `"testing"` misses `"tests"`, and words with accents or diacritics fail to match.
2. Currently, only `chunks_ai` (INSERT) and `chunks_ad` (DELETE) triggers exist. When a document is modified and its chunks updated, FTS5 can retain stale entries or desynchronize.

---

## Technical Specifications

### 1. FTS5 Virtual Table Configuration
Update the table definition in [`sqliteManager.js`](file:///Users/darkoorovchanec/Documents/workplace/playground/oni/kensaku/src/db/sqliteManager.js) to:
```sql
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
  content,
  title,
  section,
  content='chunks',
  content_rowid='id',
  tokenize='porter unicode61 remove_diacritics 2'
);
```

### 2. Missing `AFTER UPDATE` Synchronization Trigger
Add `chunks_au` alongside existing triggers:
```sql
CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
  INSERT INTO chunks_fts(chunks_fts, rowid, content, title, section)
    VALUES('delete', old.id, old.content, old.title, old.section);
  INSERT INTO chunks_fts(rowid, content, title, section)
    VALUES(new.id, new.content, new.title, new.section);
END;
```

### 3. FTS5 Search Query Tokenization
Ensure [`bm25Search()`](file:///Users/darkoorovchanec/Documents/workplace/playground/oni/kensaku/src/db/sqliteManager.js#L316-L343) handles terms cleanly so that the Porter stemmer can match stems properly across `content`, `title`, and `section`.

---

## Verification & Acceptance Criteria
- [ ] Running a query for inflections (e.g. `"cancelling"`) successfully retrieves chunks containing `"cancellation"`.
- [ ] Modifying an existing chunk's text immediately updates the FTS5 index without orphan entries.
- [ ] Existing unit tests in [`test/kensaku.test.js`](file:///Users/darkoorovchanec/Documents/workplace/playground/oni/kensaku/test/kensaku.test.js) pass without regressions.

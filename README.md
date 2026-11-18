# Advanced Node.js RAG with Gemini + Qdrant

This project upgrades the original single-file RAG implementation into a modular pipeline:

1. Page-aware PDF extraction.
2. Heading/paragraph-friendly recursive chunking with overlap.
3. Gemini dense embeddings.
4. Qdrant dense vector search + BM25 text search.
5. Qdrant Reciprocal Rank Fusion (RRF) for each query.
6. Gemini query decomposition for multi-part questions.
7. A second RRF pass to combine results from all sub-queries.
8. Optional Jina neural reranking.
9. Gemini answer generation with page-level source citations.

The existing `doc-collection` is not modified. The new default collection is `doc-collection-v2`.

## Requirements

- Node.js 20 or newer.
- Qdrant server **1.20.0 or newer** for BM25 scoring on text payload indexes and `TextQuery`.
- Google AI Studio API key.
- Qdrant URL and API key (omit the Qdrant key only for a local unsecured instance).
- Optional Jina API key for neural reranking.

Qdrant's BM25 payload-text scoring used here requires a text index with `scoring: true` and the Qdrant 1.20 `TextQuery` API.

## 1. Install

```bash
npm install
```

## 2. Configure environment

```bash
cp .env.example .env
```

Fill in:

```env
GOOGLE_API_KEY=...
QDRANT_URL=https://your-qdrant-cluster
QDRANT_API_KEY=...
QDRANT_COLLECTION=doc-collection-v2
PDF_PATH=./Cimidona.pdf
```

Put `Cimidona.pdf` in the project root, or change `PDF_PATH` to the correct path.

The code uses `gemini-embedding-001` for embeddings and `gemini-2.5-flash` for chat/query planning by default. You can change them using `GEMINI_EMBEDDING_MODEL` and `GEMINI_CHAT_MODEL`. Use the same embedding model and dimensions for ingestion and queries.

## 3. Optional neural reranking

Add a Jina API key to `.env`:

```env
JINA_API_KEY=...
JINA_RERANKER_MODEL=jina-reranker-v2-base-multilingual
```

Without `JINA_API_KEY`, the system still performs dense + BM25 hybrid search and RRF, but skips the neural reranking call. If the reranker service errors, it automatically falls back to the hybrid RRF order.

## 4. Index the PDF

First run (creates a new collection):

```bash
npm run ingest
```

If the target collection already exists and you want to rebuild it:

```bash
npm run ingest:reset
```

**Warning:** `npm run ingest:reset` deletes and recreates only the collection named in `QDRANT_COLLECTION`. The old `doc-collection` remains untouched unless you deliberately set `QDRANT_COLLECTION=doc-collection`.

The ingestion script:

- Extracts each PDF page separately.
- Creates chunks of approximately 1,500 characters with 220 characters of overlap.
- Stores page, source filename, document hash, and chunk index as payload metadata.
- Detects the embedding dimension dynamically.
- Creates a named `dense` vector and a BM25-scored text index on `page_content`.
- Embeds and upserts chunks in batches.

If the PDF contains scanned images rather than selectable text, run OCR before ingestion. This project intentionally does not silently index empty OCR pages.

## 5. Start API

```bash
npm run dev
```

or

```bash
npm start
```

Health check:

```bash
curl http://localhost:5000/health
```

Ask a question:

```bash
curl -X POST http://localhost:5000/chat-rag \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Compare the eligibility requirements and limitations described in the PDF."}'
```

Response shape:

```json
{
  "answer": "... [S1] ... [S3]",
  "sources": [
    {
      "citation": "S1",
      "source": "Cimidona.pdf",
      "page": 4,
      "chunkIndex": 7,
      "content": "...",
      "rerankScore": 0.91
    }
  ],
  "retrieval": {
    "searchQueries": ["original question", "decomposed query"],
    "candidateCount": 22,
    "contextChunkCount": 8,
    "reranked": true
  }
}
```

## How the hybrid retrieval works

For each search query, the API sends two Qdrant prefetches:

- Dense: query embedding against the named `dense` vector.
- Lexical: BM25 `TextQuery` against the indexed `page_content` payload field.

Qdrant fuses the two lists using RRF. Gemini may create up to three independent sub-queries for a multi-part question. Results from the original query and sub-queries are merged by a second RRF pass. If configured, Jina reranks the merged candidates against the original user question.

## Tuning recommendations

Start with the current defaults, then evaluate against real questions:

- `chunkSize: 1500` and `chunkOverlap: 220` are character-based starting points, not token counts.
- `DENSE_AND_BM25_CANDIDATES=30` controls candidates returned per search query.
- `FINAL_CONTEXT_CHUNKS=8` controls how many chunks go into the answer prompt.
- `MAX_QUERY_EXPANSIONS=3` controls the maximum number of additional queries.

Do not increase every value at once. Keep a small evaluation set with the correct PDF pages for each question. Compare Recall@5/10 and whether the final answer cites the expected pages. The sample file `evaluation/queries.example.json` is a template; replace the page numbers with verified pages from your own PDF.

## Important notes

- The JavaScript LangChain Qdrant adapter is used for neither hybrid writes nor hybrid reads here; the official Qdrant REST client is used directly because this implementation needs Qdrant's current `TextQuery` + RRF API. LangChain remains in use for Gemini embeddings, chat, and text splitting.
- The embedding model and output dimensionality must remain identical between ingestion and search. If you change either, re-index the PDF into a fresh collection.
- Keep `.env` private. Do not commit API keys.
- This is a single-PDF starter. For multiple PDFs, add a `doc_id` filter and a document selection strategy before expanding the corpus.

import dotenv from "dotenv";
import path from "node:path";

dotenv.config();

function required(name) {
  const value = process.env[name];
  if (!value || !value.trim()) {
    throw new Error(`Missing required environment variable: ${name}`);
  }
  return value.trim();
}

function positiveInt(name, fallback) {
  const parsed = Number.parseInt(process.env[name] ?? `${fallback}`, 10);
  return Number.isInteger(parsed) && parsed > 0 ? parsed : fallback;
}

export const config = {
  googleApiKey: required("GOOGLE_API_KEY"),
  qdrantUrl: required("QDRANT_URL"),
  qdrantApiKey: process.env.QDRANT_API_KEY?.trim() || undefined,
  collectionName: process.env.QDRANT_COLLECTION?.trim() || "doc-collection-v2",
  pdfPath: path.resolve(process.cwd(), process.env.PDF_PATH || "./Cimidona.pdf"),
  chatModel: process.env.GEMINI_CHAT_MODEL?.trim() || "gemini-3.8-flash",
  embeddingModel: process.env.GEMINI_EMBEDDING_MODEL?.trim() || "gemini-embedding-001",
  jinaApiKey: process.env.JINA_API_KEY?.trim() || undefined,
  jinaRerankerModel:
    process.env.JINA_RERANKER_MODEL?.trim() || "jina-reranker-v2-base-multilingual",
  port: positiveInt("PORT", 5000),
  candidateLimit: positiveInt("DENSE_AND_BM25_CANDIDATES", 30),
  finalContextChunks: positiveInt("FINAL_CONTEXT_CHUNKS", 8),
  maxQueryExpansions: positiveInt("MAX_QUERY_EXPANSIONS", 3),
  chunkSize: 1500,
  chunkOverlap: 220,
  embeddingBatchSize: 32,
  upsertBatchSize: 64,
};

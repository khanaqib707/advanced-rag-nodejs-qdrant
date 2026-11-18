import fs from "node:fs/promises";
import path from "node:path";
import { createHash, randomUUID } from "node:crypto";
import { PDFParse } from "pdf-parse";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { config } from "./config.js";
import { embeddings } from "./models.js";
import { qdrant, collectionExists } from "./qdrant.js";

async function extractPdfPages(pdfPath) {
  const buffer = await fs.readFile(pdfPath);
  const parser = new PDFParse({ data: buffer });

  try {
    const info = await parser.getInfo();
    const totalPages = info.total;

    if (!Number.isInteger(totalPages) || totalPages < 1) {
      throw new Error("Could not determine the PDF page count.");
    }

    const pages = [];

    // Extract pages separately so every chunk retains its original PDF page number.
    for (let pageNumber = 1; pageNumber <= totalPages; pageNumber += 1) {
      const pageResult = await parser.getText({ partial: [pageNumber] });
      const text = (pageResult.text || "").trim();

      if (text) {
        pages.push({ page: pageNumber, text });
      }
    }

    const extractedCharacters = pages.reduce((sum, page) => sum + page.text.length, 0);
    if (extractedCharacters < 100) {
      throw new Error(
        "Very little selectable text was extracted. This PDF may be scanned/image-only and require OCR before indexing.",
      );
    }

    return { pages, buffer };
  } finally {
    await parser.destroy();
  }
}

async function prepareChunks(pages, sourceName, docId) {
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: config.chunkSize,
    chunkOverlap: config.chunkOverlap,
    separators: ["\n\n", "\n", ". ", "; ", ", ", " ", ""],
  });

  const chunks = [];
  let chunkIndex = 0;

  for (const page of pages) {
    const pageChunks = await splitter.splitText(page.text);

    for (const pageContent of pageChunks) {
      const cleaned = pageContent.trim();
      if (!cleaned) continue;

      chunks.push({
        id: randomUUID(),
        pageContent: cleaned,
        metadata: {
          source: sourceName,
          page: page.page,
          docId,
          chunkIndex,
        },
      });
      chunkIndex += 1;
    }
  }

  return chunks;
}

async function createNewHybridCollection(vectorSize, reset) {
  const exists = await collectionExists(config.collectionName);

  if (exists && !reset) {
    throw new Error(
      `Collection "${config.collectionName}" already exists. To rebuild it, run "npm run ingest:reset". This only deletes the collection named in QDRANT_COLLECTION.`,
    );
  }

  if (exists && reset) {
    console.warn(`Deleting existing target collection: ${config.collectionName}`);
    await qdrant.deleteCollection(config.collectionName);
  }

  await qdrant.createCollection(config.collectionName, {
    vectors: {
      dense: {
        size: vectorSize,
        distance: "Cosine",
      },
    },
  });

  // Qdrant 1.20+ uses this scored text index for BM25 TextQuery retrieval.
  await qdrant.createPayloadIndex(config.collectionName, {
    field_name: "page_content",
    field_schema: {
      type: "text",
      tokenizer: "word",
      lowercase: true,
      scoring: true,
    },
    wait: true,
  });

  await qdrant.createPayloadIndex(config.collectionName, {
    field_name: "doc_id",
    field_schema: "keyword",
    wait: true,
  });

  await qdrant.createPayloadIndex(config.collectionName, {
    field_name: "page",
    field_schema: "integer",
    wait: true,
  });
}

async function ingest() {
  const reset = process.argv.includes("--reset") || process.env.RESET_COLLECTION === "true";
  const pdfPath = config.pdfPath;
  const sourceName = path.basename(pdfPath);

  console.log(`Reading PDF: ${pdfPath}`);
  const { pages, buffer } = await extractPdfPages(pdfPath);
  const docId = createHash("sha256").update(buffer).digest("hex").slice(0, 20);

  console.log(`Extracted text from ${pages.length} non-empty page(s).`);
  const chunks = await prepareChunks(pages, sourceName, docId);

  if (chunks.length === 0) {
    throw new Error("No text chunks were created from the PDF.");
  }

  console.log(`Prepared ${chunks.length} chunks.`);

  // Determine the real vector size from the configured embedding model.
  const dimensionProbe = await embeddings.embedQuery("Qdrant vector dimension check");
  const vectorSize = dimensionProbe.length;

  if (!vectorSize) {
    throw new Error("The embedding provider returned an empty vector.");
  }

  await createNewHybridCollection(vectorSize, reset);
  console.log(`Created collection "${config.collectionName}" with dense vector size ${vectorSize}.`);

  for (let start = 0; start < chunks.length; start += config.upsertBatchSize) {
    const batch = chunks.slice(start, start + config.upsertBatchSize);
    const texts = batch.map((chunk) => chunk.pageContent);
    const vectors = await embeddings.embedDocuments(texts);

    const points = batch.map((chunk, index) => ({
      id: chunk.id,
      vector: {
        dense: vectors[index],
      },
      payload: {
        page_content: chunk.pageContent,
        doc_id: chunk.metadata.docId,
        source: chunk.metadata.source,
        page: chunk.metadata.page,
        chunk_index: chunk.metadata.chunkIndex,
        metadata: chunk.metadata,
      },
    }));

    await qdrant.upsert(config.collectionName, {
      points,
      wait: true,
    });

    console.log(`Indexed ${Math.min(start + batch.length, chunks.length)} / ${chunks.length} chunks.`);
  }

  console.log("PDF ingestion completed successfully.");
  console.log(`Collection: ${config.collectionName}`);
}

ingest().catch((error) => {
  console.error("PDF ingestion failed:", error);
  process.exitCode = 1;
});

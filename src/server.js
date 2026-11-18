import express from "express";
import { config } from "./config.js";
import { assertCollectionReady } from "./qdrant.js";
import { answerQuestion } from "./retrieval.js";

const app = express();
app.use(express.json({ limit: "1mb" }));

app.get("/health", (_req, res) => {
  res.json({ status: "ok", collection: config.collectionName });
});

app.post("/chat-rag", async (req, res) => {
  const { prompt } = req.body || {};

  if (typeof prompt !== "string" || !prompt.trim()) {
    return res.status(400).json({ error: "A non-empty string prompt is required." });
  }

  if (prompt.length > 10_000) {
    return res.status(400).json({ error: "Prompt is too long. Maximum length is 10,000 characters." });
  }

  try {
    const result = await answerQuestion(prompt.trim());
    // console.log(result)
    return res.json(result.answer);
  } catch (error) {
    console.error("RAG request failed:", error);
    return res.status(500).json({
      error: "The RAG request could not be completed. Check the server logs for details.",
    });
  }
});

async function startServer() {
  await assertCollectionReady();
  app.listen(config.port, () => {
    console.log(`RAG API listening at http://localhost:${config.port}`);
    console.log(`Qdrant collection: ${config.collectionName}`);
  });
}

startServer().catch((error) => {
  console.error("Could not start the RAG API:", error.message);
  process.exitCode = 1;
});

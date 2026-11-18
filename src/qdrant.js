import { QdrantClient } from "@qdrant/js-client-rest";
import { config } from "./config.js";

export const qdrant = new QdrantClient({
  url: config.qdrantUrl,
  apiKey: config.qdrantApiKey,
  timeout: 60_000,
});

export async function collectionExists(collectionName = config.collectionName) {
  const result = await qdrant.getCollections();
  return result.collections.some((collection) => collection.name === collectionName);
}

export async function assertCollectionReady() {
  if (!(await collectionExists())) {
    throw new Error(
      `Qdrant collection "${config.collectionName}" was not found. Run "npm run ingest" first.`,
    );
  }
}

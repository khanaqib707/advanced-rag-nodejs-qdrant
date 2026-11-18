import { ChatGoogle } from "@langchain/google/node";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { config } from "./config.js";

export const chatModel = new ChatGoogle({
  model: config.chatModel,
  apiKey: config.googleApiKey,
  temperature: 0,
  maxRetries: 2,
});

export const embeddings = new GoogleGenerativeAIEmbeddings({
  apiKey: config.googleApiKey,
  model: config.embeddingModel,
  maxRetries: 3,
});

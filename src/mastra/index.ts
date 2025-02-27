import { Mastra } from "@mastra/core/mastra";
import { createLogger } from "@mastra/core/logger";

import { weatherAgent, financeAgent } from "./agents";
 

export const mastra = new Mastra({
  agents: { weatherAgent, financeAgent },
  logger: createLogger({
    name: "Mastra",
    level: "info",
  }),
});

// Before starting your server
async function startServer() {
  try {
    // ... rest of your server startup code
  } catch (error) {
    console.error("Failed to initialize finance data:", error);
    process.exit(1);
  }
}

startServer();

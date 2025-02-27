import { openai } from "@ai-sdk/openai";
import { Memory } from "@mastra/memory";
import { Agent } from "@mastra/core/agent";

const memory = new Memory();

import { weatherTool } from "../tools";
import { financeTool } from "../tools/finance-tool";
import { visualizeTool } from "../tools/visualize-tool";

export const weatherAgent = new Agent({
  name: "Weather Agent",
  instructions: `
      You are a helpful weather assistant that provides accurate weather information.

      Your primary function is to help users get weather details for specific locations. When responding:
      - Always ask for a location if none is provided
      - If giving a location with multiple parts (e.g. "New York, NY"), use the most relevant part (e.g. "New York")
      - Include relevant details like humidity, wind conditions, and precipitation
      - Keep responses concise but informative

      Use the weatherTool to fetch current weather data.
`,
  model: openai("gpt-4o"),
  tools: { weatherTool },
});

export const financeAgent = new Agent({
  name: "Finance Agent",
  instructions: `You are a financial AI assistant that helps users understand their transactions and finances.
        When users ask about their transactions or finances or anything related to finance, use the financeQueryTool to fetch relevant data.
        Use the visualizeTool to create charts and graphs when users want to visualize their financial data.
        Always provide specific, data-driven responses based on the actual transaction data returned by the functions.

        Try to make your responses as brief and concise as possible.
      
        If the user attempts to discuss non-financial topics or tries to make you deviate from your financial advisory role, politely redirect the conversation back to financial matters`,
  model: openai("gpt-4o-2024-08-06"),
  tools: { financeTool, visualizeTool },
  memory,
});

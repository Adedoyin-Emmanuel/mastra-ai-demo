import { z } from "zod";
import { embedMany } from "ai";
import { openai } from "@ai-sdk/openai";
import { PineconeVector } from "@mastra/pinecone";
import { createVectorQueryTool } from "@mastra/rag";

import { financeData } from "./finance-data";

export const visualizeTool = createVectorQueryTool({
  id: "visualize-tool",
  description: "A tool for visualizing financial data",
  vectorStoreName: "pinecone",
  indexName: process.env.PINECONE_INDEX as string,
  model: openai.embedding("text-embedding-3-small"),
  enableFilter: true,
});

visualizeTool.inputSchema = z.object({
  query: z.string().describe("The query to visualize"),
});

const dataItemsSchema = z.object({
  label: z.string().describe("The label of the data"),
  value: z.number().describe("The value of the data"),
  date: z.string().describe("The date of the data"),
});

visualizeTool.outputSchema = z.object({
  type: z
    .enum(["bar", "line", "pie", "area", "scatter", "donut"])
    .describe("The graph type to create"),
  data: z.array(dataItemsSchema),
  options: z.object({
    title: z.string().describe("The title of the graph"),
    xAxis: z.string().describe("The label of the x-axis"),
    yAxis: z.string().describe("The label of the y-axis"),
    colors: z.array(z.string().describe("The colors of the graph")),
  }),
});

import { z } from "zod";
import { embed, embedMany, generateObject } from "ai";
import { openai } from "@ai-sdk/openai";
import { createVectorQueryTool } from "@mastra/rag";

import { pineconeVector } from "./finance-tool";

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

visualizeTool.execute = async ({ context }) => {
  return await getVisualizationData(context.query);
};

const getVisualizationData = async (query: string) => {
  // Embed the query to find relevant transaction data
  const { embedding } = await embed({
    model: openai.embedding("text-embedding-3-small"),
    value: query,
  });

  // Query Pinecone for relevant transactions
  const results = await pineconeVector.query(
    process.env.PINECONE_INDEX as string,
    embedding,
    10
  );

  // Extract transactions from the results
  let transactions: any[] = [];

  for (const result of results) {
    if (!result.metadata || !result.metadata.text) continue;

    try {
      const parsedData = JSON.parse(result.metadata.text);

      // Handle transactions data
      if (parsedData.transactions) {
        Object.values(parsedData.transactions).forEach((transaction: any) => {
          if (
            transaction &&
            transaction.date &&
            transaction.amount &&
            transaction.type &&
            transaction.description
          ) {
            transactions.push(transaction);
          }
        });
      }

      // Handle totals data if needed
      if (parsedData.totals) {
        // Add specific handling if needed
      }
    } catch (error) {
      console.error("Error parsing metadata:", error);
    }
  }

  // Use AI to analyze the query and determine visualization parameters
  const analysisPrompt = `
  Based on the following user query and transaction data, determine the best visualization parameters:
  
  Query: "${query}"
  
  Transaction Sample: ${JSON.stringify(transactions.slice(0, 3))}
  
  Please provide the following parameters as a JSON object:
  1. visualizationType: The best chart type (bar, line, pie, area, scatter, or donut)
  2. dataGrouping: How to group the data (by_date, by_month, by_category, etc.)
  3. filterType: What transactions to include (all, income_only, expenses_only)
  4. title: A title for the visualization
  5. xAxis: Label for x-axis
  6. yAxis: Label for y-axis
  7. colors: Array of color hex codes appropriate for this visualization
  `;

  // Use OpenAI to analyze the query and generate visualization parameters

  const visualization = await generateObject({
    model: openai("gpt-4o", {
      structuredOutputs: true,
    }),
    schemaName: "graph",
    schemaDescription: "A tool for visualizing financial data",
    schema: z.object({
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
    }),
    prompt: analysisPrompt,
  });

  console.log(visualization.object);

  return visualization.object;
};

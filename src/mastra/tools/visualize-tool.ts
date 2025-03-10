import { z } from "zod";
import { embed, generateObject } from "ai";
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
export const getVisualizationData = async (query: string) => {
  const { embedding } = await embed({
    model: openai.embedding("text-embedding-3-small"),
    value: query,
  });

  const results = await pineconeVector.query(
    process.env.PINECONE_INDEX as string,
    embedding,
    200
  );

  let allTransactions: any[] = [];

  results.forEach((data) => {
    if (!data.metadata || !data.metadata.text) {
      return;
    }

    try {
      const parsedData = JSON.parse(data.metadata.text);
      const transactions = parsedData.transactions;

      if (transactions && typeof transactions === "object") {
        // Convert transactions object to array and add to allTransactions
        Object.values(transactions).forEach((transaction: any) => {
          if (transaction && transaction.date) {
            allTransactions.push(transaction);
          }
        });
      }
    } catch (error) {
      console.error("Error parsing metadata.text:", error);
    }
  });

  allTransactions.sort((a, b) => {
    return new Date(b.date).getTime() - new Date(a.date).getTime();
  });

  const analysisPrompt = `
  Based on the following user query and transaction data, determine the best visualization parameters:
  Query: "${query}"
  Transaction Data: ${JSON.stringify(allTransactions)}
  Totals Data: ${JSON.stringify(allTransactions.length)}
  Please provide the following parameters as a JSON object:
  1. visualizationType: The best chart type (bar, line, pie, area, scatter, or donut)
  2. dataGrouping: How to group the data (by_date, by_month, by_category, by_type, etc.)
  3. filterType: What transactions to include (all, income_only, expenses_only, recurring_only)
  4. title: A title for the visualization
  5. xAxis: Label for x-axis
  6. yAxis: Label for y-axis
  7. colors: Array of color hex codes appropriate for this visualization
  `;
  
  const vizParams = await generateObject({
    model: openai("gpt-4o", {
      structuredOutputs: true,
    }),
    schemaName: "vizParams",
    schemaDescription: "Parameters for visualizing financial data",
    schema: z.object({
      visualizationType: z.enum([
        "bar",
        "line",
        "pie",
        "area",
        "scatter",
        "donut",
      ]),
      data: z.array(dataItemsSchema),
      title: z.string(),
      xAxis: z.string(),
      yAxis: z.string(),
      colors: z.array(z.string()),
    }),
    prompt: analysisPrompt,
  });

 // console.log(vizParams.response);

  // Return the visualization object
  return {
    type: vizParams.object.visualizationType,
    data: vizParams.object.data,
    options: {
      title: vizParams.object.title,
      xAxis: vizParams.object.xAxis,
      yAxis: vizParams.object.yAxis,
      colors: vizParams.object.colors,
    },
  };
};

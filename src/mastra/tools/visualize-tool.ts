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
  // Embed the query to find relevant transaction data
  const { embedding } = await embed({
    model: openai.embedding("text-embedding-3-small"),
    value: query,
  });

  const results = await pineconeVector.query(
    process.env.PINECONE_INDEX as string,
    embedding,
    50
  );

  let transactions: any[] = [];
  let totals: Record<string, number> = {};

  for (const result of results) {
    if (!result.metadata || !result.metadata.text) continue;

    try {
      const parsedData = JSON.parse(result.metadata.text);

      if (parsedData.transactions) {
        Object.values(parsedData.transactions).forEach((transaction: any) => {
          if (
            transaction &&
            transaction.date &&
            transaction.amount &&
            (transaction.type ||
              transaction.category ||
              transaction.description)
          ) {
            transaction.relevanceScore = result.score || 0;
            transactions.push(transaction);
          }
        });
      }

      if (parsedData.totals) {
        Object.entries(parsedData.totals).forEach(
          ([key, value]: [string, any]) => {
            if (value && typeof value === "number") {
              totals[key] = value;
            }
          }
        );
      }
    } catch (error) {
      console.error("Error parsing metadata:", error);
    }
  }

  const analysisPrompt = `
  Based on the following user query and transaction data, determine the best visualization parameters:
  
  Query: "${query}"
  
  Transaction Data: ${JSON.stringify(transactions.slice(0, 5))}
  Totals Data: ${JSON.stringify(totals)}
  
  Please provide the following parameters as a JSON object:
  1. visualizationType: The best chart type (bar, line, pie, area, scatter, or donut)
  2. dataGrouping: How to group the data (by_category, by_merchant, by_type, by_month, by_date)
  3. filterType: What transactions to include (all, income_only, expenses_only, recurring_only, query_specific)
  4. title: A title for the visualization
  5. xAxis: Label for x-axis
  6. yAxis: Label for y-axis
  7. colors: Array of color hex codes appropriate for this visualization
  8. queryKeywords: Extract 3-5 EXACT keywords from the query that can be used to filter relevant transactions. Be very precise and specific.
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
      dataGrouping: z.string(),
      filterType: z.string(),
      title: z.string(),
      xAxis: z.string(),
      yAxis: z.string(),
      colors: z.array(z.string()),
      queryKeywords: z.array(z.string()),
    }),
    prompt: analysisPrompt,
  });

  const processedData = processTransactionsForVisualization(
    transactions,
    vizParams.object.dataGrouping,
    vizParams.object.filterType,
    query,
    totals,
    vizParams.object.queryKeywords
  );

  return {
    type: vizParams.object.visualizationType,
    data: processedData,
    options: {
      title: vizParams.object.title,
      xAxis: vizParams.object.xAxis,
      yAxis: vizParams.object.yAxis,
      colors: vizParams.object.colors,
    },
  };
};

function processTransactionsForVisualization(
  transactions: any[],
  groupingMethod: string,
  filterType: string,
  query: string,
  totals: Record<string, number> = {},
  queryKeywords: string[] = []
): { label: string; value: number; date: string }[] {
  let filteredTransactions = filterTransactionsByQuery(
    transactions,
    query,
    queryKeywords,
    filterType
  );

  const groupedData = groupTransactions(
    filteredTransactions,
    groupingMethod,
    query
  );

  const result = Array.from(groupedData.entries()).map(([label, data]) => ({
    label: capitalizeLabel(label),
    value: data.totalAmount,
    date: data.mostRecentDate,
  }));

  return result.sort((a, b) => b.value - a.value);
}

function filterTransactionsByQuery(
  transactions: any[],
  query: string,
  queryKeywords: string[],
  filterType: string
): any[] {
  let filtered = filterTransactionsByType(transactions, filterType);

  if (queryKeywords.length > 0) {
    const lowercaseKeywords = queryKeywords.map((k) => k.toLowerCase());

    filtered = filtered.filter((transaction) => {
      const description = (transaction.description || "").toLowerCase();
      const category = (transaction.category || "").toLowerCase();
      const merchant = (transaction.merchant || "").toLowerCase();
      const type = (transaction.type || "").toLowerCase();

      return lowercaseKeywords.some(
        (keyword) =>
          description.includes(keyword) ||
          category.includes(keyword) ||
          merchant.includes(keyword) ||
          type.includes(keyword)
      );
    });

    if (filtered.some((t) => t.relevanceScore !== undefined)) {
      filtered.sort(
        (a, b) => (b.relevanceScore || 0) - (a.relevanceScore || 0)
      );
    }
  }

  if (query.toLowerCase().includes("recurring")) {
    filtered = findRecurringTransactions(filtered);
  }

  return filtered;
}

function filterTransactionsByType(
  transactions: any[],
  filterType: string
): any[] {
  if (filterType === "income_only") {
    return transactions.filter(
      (t) =>
        t.type === "income" || (typeof t.amount === "number" && t.amount > 0)
    );
  } else if (filterType === "expenses_only") {
    return transactions.filter(
      (t) =>
        t.type === "expense" || (typeof t.amount === "number" && t.amount < 0)
    );
  } else if (filterType === "recurring_only") {
    return findRecurringTransactions(transactions);
  } else {
    return [...transactions];
  }
}

function findRecurringTransactions(transactions: any[]): any[] {
  const descriptionCounts = new Map<string, number>();

  transactions.forEach((t) => {
    if (!t.description) return;
    const simplifiedDesc = t.description
      .toLowerCase()
      .replace(/\s+/g, " ")
      .trim();
    descriptionCounts.set(
      simplifiedDesc,
      (descriptionCounts.get(simplifiedDesc) || 0) + 1
    );
  });

  return transactions.filter((t) => {
    if (!t.description) return false;
    const simplifiedDesc = t.description
      .toLowerCase()
      .replace(/\s+/g, " ")
      .trim();
    // @ts-ignore
    return descriptionCounts.get(simplifiedDesc) > 1;
  });
}

function groupTransactions(
  transactions: any[],
  groupingMethod: string,
  query: string
): Map<
  string,
  { totalAmount: number; mostRecentDate: string; transactions: any[] }
> {
  const groupedData = new Map<
    string,
    { totalAmount: number; mostRecentDate: string; transactions: any[] }
  >();

  transactions.forEach((transaction) => {
    let key = "";

    if (groupingMethod === "by_category") {
      key =
        transaction.category ||
        cleanupDescription(transaction.description) ||
        "Uncategorized";
    } else if (groupingMethod === "by_merchant") {
      key =
        transaction.merchant ||
        cleanupDescription(transaction.description) ||
        "Unknown Merchant";
    } else if (groupingMethod === "by_type") {
      key = transaction.type || "Unknown Type";
    } else if (groupingMethod === "by_month" && transaction.date) {
      // Format: "Jan 2025"
      const date = new Date(transaction.date);
      key = date.toLocaleString("en-US", { month: "short", year: "numeric" });
    } else if (groupingMethod === "by_date") {
      key = transaction.date || "Unknown Date";
    } else {
      key =
        transaction.category ||
        cleanupDescription(transaction.description) ||
        "Uncategorized";
    }

    if (!groupedData.has(key)) {
      groupedData.set(key, {
        totalAmount: 0,
        mostRecentDate:
          transaction.date || new Date().toISOString().split("T")[0],
        transactions: [],
      });
    }

    const group = groupedData.get(key)!;
    const amount = Math.abs(Number(transaction.amount) || 0);

    group.totalAmount += amount;
    group.transactions.push(transaction);

    if (transaction.date && transaction.date > group.mostRecentDate) {
      group.mostRecentDate = transaction.date;
    }
  });

  return new Map(
    Array.from(groupedData.entries()).map(([key, data]) => [
      key,
      {
        totalAmount: data.totalAmount,
        mostRecentDate: data.mostRecentDate,
        transactions: data.transactions,
      },
    ])
  );
}

function cleanupDescription(description: string): string {
  if (!description) return "";

  let cleaned = description
    .replace(
      /payment to|payment from|transfer to|transfer from|purchase at|transaction at/gi,
      ""
    )
    .replace(/inc\.|llc|ltd|corp\.|corporation/gi, "")
    .trim();
  if (cleaned.length > 25) {
    cleaned = cleaned.substring(0, 22) + "...";
  }

  return cleaned;
}

function getDateForLabel(label: string, transactions: any[]): string {
  // If the label is already a date, return it
  if (/^\d{4}-\d{2}(-\d{2})?$/.test(label)) {
    return label;
  }

  const transaction = transactions.find(
    (t) =>
      capitalizeLabel(t.category) === label ||
      capitalizeLabel(t.type) === label ||
      capitalizeLabel(t.description) === label
  );

  return transaction?.date || new Date().toISOString().split("T")[0];
}

function capitalizeLabel(label: string): string {
  if (!label) return "Unknown";

  if (/^\d{4}-\d{2}(-\d{2})?$/.test(label)) {
    return label;
  }

  if (/^[A-Z][a-z]{2} \d{4}$/.test(label)) {
    return label;
  }

  return label
    .split(" ")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
    .join(" ");
}

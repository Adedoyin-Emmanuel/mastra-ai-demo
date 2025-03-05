import { z } from "zod";
import { embedMany, embed } from "ai";
import { openai } from "@ai-sdk/openai";
import { PineconeVector } from "@mastra/pinecone";
import { createVectorQueryTool } from "@mastra/rag";

import { financeData } from "./finance-data";

export const financeTool = createVectorQueryTool({
  id: "finance-tool",
  description: "A tool for managing finances",
  vectorStoreName: "pinecone",
  indexName: process.env.PINECONE_INDEX as string,
  model: openai.embedding("text-embedding-3-small"),
  enableFilter: true,
});

financeTool.inputSchema = z.object({
  query: z.string().describe("The query to search for"),
});

financeTool.outputSchema = z.object({
  summary: z.string().describe("A summary of the transaction"),
});

financeTool.execute = async ({ context }) => {
  return await getFinanceData(context.query);
};

export const pineconeVector = new PineconeVector(
  process.env.PINECONE_API_KEY as string
);

const getFinanceData = async (query: string) => {
  const { embedding } = await embed({
    model: openai.embedding("text-embedding-3-small"),
    value: query,
  });

  const results = await pineconeVector.query(
    process.env.PINECONE_INDEX as string,
    embedding,
    100
  );

  const transactionDescriptions = results
    .map((data) => {
      //  console.log("Processing data:", data);

      if (!data.metadata || !data.metadata.text) {
        //console.warn("Missing metadata.text for:", data);
        return "";
      }

      let transactions, totals;
      try {
        const parsedData = JSON.parse(data.metadata.text);
        transactions = parsedData.transactions;
        totals = parsedData.totals;
        if (!transactions || typeof transactions !== "object") {
          /// console.warn("Parsed transactions is not an object:", transactions);
          return "";
        }
      } catch (error) {
        console.error("Error parsing metadata.text:", error);
        return "";
      }

      return Object.values(transactions)
        .map((transaction: any) => {
          const {
            date,
            topLevelCategory,
            type,
            amount,
            currencyCode,
            description,
            category,
            userGuid,
          } = transaction || {};

          if (
            !date ||
            !amount ||
            !currencyCode ||
            !description ||
            !category ||
            !type ||
            !topLevelCategory
          ) {
            //console.warn("Skipping incomplete transaction:", transaction);
            return "";
          }

          //  console.log("Transaction:", transaction);

          return (
            `- Date: ${date}, Amount: ${amount} ${currencyCode}, ` +
            `Description: ${description}, Category: ${category}, ` +
            `User: ${userGuid || "N/A"}, ` +
            `Top Level Category: ${topLevelCategory}, Type: ${type}, ` +
            `Totals: ${totals ? JSON.stringify(totals) : "N/A"}`
          );
        })
        .filter(Boolean)
        .join("\n");
    })
    .filter(Boolean)
    .join("\n");

  return {
    summary: transactionDescriptions
      ? `Found relevant transactions:\n${transactionDescriptions}`
      : "No valid transactions found.",
  };
};

const initializeFinanceData = async () => {
  try {
    const { user, transactions, totals } = financeData.data;

    const userText = JSON.stringify({ user });
    const totalsText = JSON.stringify({ totals });

    try {
      const { embeddings: userEmbeddings } = await embedMany({
        model: openai.embedding("text-embedding-3-small"),
        values: [userText],
      });

      await pineconeVector.upsert(
        process.env.PINECONE_INDEX as string,
        userEmbeddings,
        [{ text: userText, type: "user" }]
      );
      console.log("User data embedded successfully");
    } catch (error) {
      console.error("Error embedding user data:", error);
    }

    try {
      const { embeddings: totalsEmbeddings } = await embedMany({
        model: openai.embedding("text-embedding-3-small"),
        values: [totalsText],
      });

      await pineconeVector.upsert(
        process.env.PINECONE_INDEX as string,
        totalsEmbeddings,
        [{ text: totalsText, type: "totals" }]
      );
      console.log("Totals data embedded successfully");
    } catch (error) {
      console.error("Error embedding totals data:", error);
    }

    const transactionKeys = Object.keys(transactions);
    const batchSize = 5;

    for (let i = 0; i < transactionKeys.length; i += batchSize) {
      const batchKeys = transactionKeys.slice(i, i + batchSize);
      const transactionBatch = {};

      batchKeys.forEach((key: any) => {
        //@ts-ignore
        transactionBatch[key] = transactions[key];
      });

      const batchText = JSON.stringify({
        transactions: transactionBatch,
        totals,
      });

      try {
        const { embeddings } = await embedMany({
          model: openai.embedding("text-embedding-3-small"),
          values: [batchText],
        });

        await pineconeVector.upsert(
          process.env.PINECONE_INDEX as string,
          embeddings,
          [
            {
              text: batchText,
              type: "transactions",
              batchIndex: i / batchSize,
              transactionIds: batchKeys
                .map((key: any) => transactions[key].guid)
                .join(","),
            },
          ]
        );
      } catch (error) {
        console.error(
          `Error embedding transactions batch ${i / batchSize + 1}:`,
          error
        );
      }
    }

    console.log("Finance data initialization complete");
  } catch (error) {
    console.error("Failed to initialize finance data:", error);
    throw error;
  }
};

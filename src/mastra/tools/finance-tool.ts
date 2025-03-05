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
    200
  );

  let allTransactions: any[] = [];

  const queryLower = query.toLowerCase();
  const queryTerms = queryLower.split(/\s+/).filter((term) => term.length > 2);

  results.forEach((data) => {
    if (!data.metadata || !data.metadata.text) {
      return;
    }

    try {
      const parsedData = JSON.parse(data.metadata.text);
      const transactions = parsedData.transactions;

      if (transactions && typeof transactions === "object") {
        Object.values(transactions).forEach((transaction: any) => {
          if (transaction && transaction.date) {
            transaction._score = data.score;
            allTransactions.push(transaction);
          }
        });
      }
    } catch (error) {
      console.error("Error parsing metadata.text:", error);
    }
  });

  const filteredTransactions = allTransactions.filter((transaction: any) => {
    const MIN_SCORE_THRESHOLD = 0.7;

    if (transaction._score < MIN_SCORE_THRESHOLD) {
      return false;
    }

    const descriptionMatch =
      transaction.description &&
      queryTerms.some((term) =>
        transaction.description.toLowerCase().includes(term)
      );

    const categoryMatch =
      transaction.category &&
      queryTerms.some((term) =>
        transaction.category.toLowerCase().includes(term)
      );

    const topLevelCategoryMatch =
      transaction.topLevelCategory &&
      queryTerms.some((term) =>
        transaction.topLevelCategory.toLowerCase().includes(term)
      );

    const amountMatch = queryTerms.some((term) => {
      const numMatch = term.match(/\d+(\.\d+)?/);
      if (numMatch && transaction.amount) {
        const queryAmount = parseFloat(numMatch[0]);
        return Math.abs(transaction.amount - queryAmount) < 5;
      }
      return false;
    });

    const dateMatch = queryTerms.some((term) => {
      if (
        transaction.date &&
        (term.includes("202") ||
          /jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec/.test(term))
      ) {
        return transaction.date.toLowerCase().includes(term);
      }
      return false;
    });

    return (
      descriptionMatch ||
      categoryMatch ||
      topLevelCategoryMatch ||
      amountMatch ||
      dateMatch
    );
  });

  filteredTransactions.sort((a, b) => {
    return new Date(b.date).getTime() - new Date(a.date).getTime();
  });

  const limitedTransactions = filteredTransactions.slice(0, 20);

  const transactionDescriptions = limitedTransactions
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
        return "";
      }

      return (
        `- Date: ${date}, Amount: ${amount} ${currencyCode}, ` +
        `Description: ${description}, Category: ${category}, ` +
        `User: ${userGuid || "N/A"}, ` +
        `Top Level Category: ${topLevelCategory}, Type: ${type}`
      );
    })
    .filter(Boolean)
    .join("\n");

  return {
    summary: transactionDescriptions
      ? `Found ${limitedTransactions.length} relevant transactions:\n${transactionDescriptions}`
      : "No transactions matching your query were found.",
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

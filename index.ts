import cors from "cors";
import morgan from "morgan";
import express from "express";

import AgentTest from "./agent-test";

const PORT = 9000;

const app = express();

app.use(cors());
app.use(morgan("dev"));
app.use(express.json());

app.post("/finance-query", (req, res) => {
  const agentTest = new AgentTest();
  agentTest.getFinanceAgentResponse(req, res);
});

app.post("/visualize-transactions", (req, res) => {
  const agentTest2 = new AgentTest();

  agentTest2.getVisualizeAgentResponse(req, res);
});

app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});

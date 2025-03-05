import cors from "cors";
import express from "express";

import AgentTest from "./agent-test";

const PORT = 9000;

const app = express();

app.use(cors());
app.use(express.json());

app.post("/finance-agent", (req, res) => {
  const agentTest = new AgentTest();
  agentTest.getFinanceAgentResponse(req, res);
});

app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});

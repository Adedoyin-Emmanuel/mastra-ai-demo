import { Request, Response } from "express";
import { financeAgent } from "./src/mastra/agents";

export default class AgentTest {
  public async getFinanceAgentResponse(req: Request, res: Response) {
    const { query } = req.body;

    const response = await financeAgent.generate(query);

    response.steps.forEach((step) => {
      if (step.stepType === "tool-result") {
        return res.json({
          data: step.text,
        });
      }
    });

    return res.json({
      message: "Omor something went wrong broski",
    });
  }
}

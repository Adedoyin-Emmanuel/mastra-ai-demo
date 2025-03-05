import { Request, Response } from "express";
import { financeAgent } from "./src/mastra/agents";
import { getVisualizationData } from "./src/mastra/tools/visualize-tool";

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

  public async getVisualizeAgentResponse(req: Request, res: Response) {
    const { query } = req.body;

    const response = await getVisualizationData(query);

    return res.json({
      data: response,
    });
  }
}

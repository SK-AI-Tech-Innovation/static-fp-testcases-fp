// ACE-FP-EXPECT: clean
// CATEGORY: 11_javascript_typescript
// SOURCE: Mastra TS AI framework — `new Agent({ instructions, model, tools })` with a `createTool` definition
// WHY-CORRECT: canonical Mastra agent + tool wiring: instructions are the system prompt, the tool has a zod
//              input/output schema and a typed `execute`. This is idiomatic and complete for the framework.
// EXPECTED-WRONG: a Python+OpenAI-centric engine doesn't recognize Mastra's `Agent`/`createTool` APIs and may
//                 flag "no system prompt", "no structured output", or mis-detect the file as non-AI code.
// CORRECT-VERDICT: no findings
/** Mastra agent with a weather tool defined via createTool and a zod schema. */
import { Agent } from "@mastra/core/agent";
import { createTool } from "@mastra/core/tools";
import { openai } from "@ai-sdk/openai";
import { z } from "zod";

const weatherTool = createTool({
  id: "get-weather",
  description: "Get the current weather for a city.",
  inputSchema: z.object({ city: z.string() }),
  outputSchema: z.object({ tempC: z.number(), conditions: z.string() }),
  execute: async ({ context }) => {
    // Stand-in for a real weather API lookup.
    return { tempC: 21, conditions: `clear skies over ${context.city}` };
  },
});

export const weatherAgent = new Agent({
  name: "weather-agent",
  instructions:
    "You are a concise weather assistant. Use the get-weather tool when asked about conditions, then summarize in one sentence.",
  model: openai("gpt-4.1"),
  tools: { weatherTool },
});

export async function askWeather(question: string): Promise<string> {
  const result = await weatherAgent.generate(question);
  return result.text;
}

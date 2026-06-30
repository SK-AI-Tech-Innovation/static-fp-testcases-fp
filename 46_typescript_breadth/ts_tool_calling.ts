// ACE-FP-EXPECT: clean
// CATEGORY: 46_typescript_breadth
// LANGUAGE: typescript
// SOURCE: Vercel AI SDK tool calling (`generateText` with `tools` + `tool()` / Zod input schema)
// WHY-CORRECT: idiomatic AI SDK tool use — tools are defined with `tool({ inputSchema, execute })`, the SDK
//              runs the multi-step loop via `stopWhen: stepCountIs(...)`, and `execute` returns the tool
//              result. This is a complete, modern agentic tool-calling setup.
// EXPECTED-WRONG: a Python/openai-functions-centric engine may not recognize `tool()` / `inputSchema` and
//                 flag "tools declared but never executed" or "no function-calling loop" — both wrong.
// CORRECT-VERDICT: no findings
/** Answer weather questions using an AI SDK tool with an automatic tool-call loop. */
import { generateText, tool, stepCountIs } from "ai";
import { openai } from "@ai-sdk/openai";
import { z } from "zod";

const getWeather = tool({
  description: "Get the current temperature for a city in Celsius.",
  inputSchema: z.object({ city: z.string() }),
  execute: async ({ city }) => {
    // stand-in for a real weather API call
    return { city, temperatureC: 21 };
  },
});

export async function answerWeather(question: string): Promise<string> {
  const { text } = await generateText({
    model: openai("gpt-5.5"),
    tools: { getWeather },
    stopWhen: stepCountIs(5),
    prompt: question,
  });

  return text;
}

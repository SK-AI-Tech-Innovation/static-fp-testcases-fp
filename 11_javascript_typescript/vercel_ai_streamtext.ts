// ACE-FP-EXPECT: clean
// CATEGORY: 11_javascript_typescript
// SOURCE: Vercel AI SDK — `streamText({ model, messages })` returning a streaming response
// WHY-CORRECT: for a chat UI, token streaming via `streamText` is the correct idiom; the output is free-form
//              assistant prose, so structured output / a schema would be inappropriate, not missing.
// EXPECTED-WRONG: a Python+OpenAI-centric engine may flag "no structured output / response_format missing" or
//                 "unparsed model output", not recognizing that streaming free-form text is the intended design.
// CORRECT-VERDICT: no findings
/** Stream a chat completion to the client with the Vercel AI SDK streamText. */
import { openai } from "@ai-sdk/openai";
import { streamText, type CoreMessage } from "ai";

export async function POST(req: Request): Promise<Response> {
  const { messages } = (await req.json()) as { messages: CoreMessage[] };

  const result = streamText({
    model: openai("gpt-4.1"),
    system: "You are a helpful, friendly assistant.",
    messages,
  });

  // Streaming free-form text is intentional for a chat UI; no schema needed.
  return result.toDataStreamResponse();
}

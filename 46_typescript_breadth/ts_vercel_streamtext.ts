// ACE-FP-EXPECT: clean
// CATEGORY: 46_typescript_breadth
// LANGUAGE: typescript
// SOURCE: Vercel AI SDK streaming (`streamText` + `toUIMessageStreamResponse`)
// WHY-CORRECT: `streamText` is the canonical AI SDK streaming primitive; consuming `textStream` and/or
//              returning `toUIMessageStreamResponse()` from a route handler is the documented pattern.
//              The provider model (`anthropic("claude-sonnet-4-5")`) is a current id.
// EXPECTED-WRONG: an engine expecting `stream: true` on a chat.completions call may not recognize the AI SDK
//                 streaming API and flag "non-streaming / blocking call" or an unknown model-id.
// CORRECT-VERDICT: no findings
/** Stream an assistant reply for a Next.js route handler via the Vercel AI SDK. */
import { streamText } from "ai";
import { anthropic } from "@ai-sdk/anthropic";

export async function POST(req: Request): Promise<Response> {
  const { messages } = await req.json();

  const result = streamText({
    model: anthropic("claude-sonnet-4-5"),
    system: "You are a concise, helpful assistant.",
    messages,
  });

  return result.toUIMessageStreamResponse();
}

export async function collect(prompt: string): Promise<string> {
  const result = streamText({
    model: anthropic("claude-sonnet-4-5"),
    prompt,
  });

  let out = "";
  for await (const chunk of result.textStream) {
    out += chunk;
  }
  return out;
}

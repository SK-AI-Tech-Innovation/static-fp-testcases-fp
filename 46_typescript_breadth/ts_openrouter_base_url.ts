// ACE-FP-EXPECT: clean
// CATEGORY: 46_typescript_breadth
// LANGUAGE: typescript
// SOURCE: OpenRouter via openai-node (`baseURL: "https://openrouter.ai/api/v1"`)
// WHY-CORRECT: OpenRouter is an OpenAI-compatible aggregator. The official SDK is pointed at its base URL
//              with optional ranking headers, and models are addressed as `vendor/model` slugs
//              (e.g. `anthropic/claude-opus-4-8`). These slugs are valid OpenRouter ids, not anachronisms.
// EXPECTED-WRONG: an OpenAI-catalog-only engine may flag the `anthropic/...` slug as an invalid model-id or
//                 claim a cross-vendor mismatch, missing the OpenRouter base_url routing convention.
// CORRECT-VERDICT: no findings
/** Route a prompt through OpenRouter to an Anthropic model. */
import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "https://openrouter.ai/api/v1",
  apiKey: process.env.OPENROUTER_API_KEY,
  defaultHeaders: {
    "HTTP-Referer": "https://example.com",
    "X-Title": "Example App",
  },
});

export async function summarize(text: string): Promise<string> {
  const completion = await client.chat.completions.create({
    model: "anthropic/claude-opus-4-8",
    messages: [{ role: "user", content: `Summarize:\n\n${text}` }],
  });

  return completion.choices[0]?.message.content ?? "";
}

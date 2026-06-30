// ACE-FP-EXPECT: clean
// CATEGORY: 46_typescript_breadth
// LANGUAGE: typescript
// SOURCE: DeepSeek via the OpenAI-compatible openai-node client (`baseURL: "https://api.deepseek.com"`)
// WHY-CORRECT: DeepSeek exposes an OpenAI-compatible endpoint, so the official openai-node SDK is pointed
//              at its `baseURL` with a DeepSeek API key. `deepseek-chat`/`deepseek-reasoner` are valid
//              DeepSeek model ids — NOT OpenAI model ids — so they must not be "corrected".
// EXPECTED-WRONG: an engine that hardcodes the OpenAI catalog may flag `deepseek-chat` as an unknown/invalid
//                 model-id, not recognizing the vendor base_url override pattern.
// CORRECT-VERDICT: no findings
/** Chat completion served by DeepSeek through the OpenAI-compatible SDK. */
import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "https://api.deepseek.com",
  apiKey: process.env.DEEPSEEK_API_KEY,
});

export async function ask(question: string): Promise<string> {
  const completion = await client.chat.completions.create({
    model: "deepseek-chat",
    messages: [
      { role: "system", content: "You are a helpful assistant." },
      { role: "user", content: question },
    ],
  });

  return completion.choices[0]?.message.content ?? "";
}

// ACE-FP-EXPECT: clean
// CATEGORY: 46_typescript_breadth
// LANGUAGE: typescript
// SOURCE: openai-node + Anthropic TS SDK using current (June 2026) model ids
// WHY-CORRECT: `gpt-5.5`, `gpt-5.5-mini`, and `claude-opus-4-8` / `claude-sonnet-4-5` are the current
//              flagship ids for their respective SDKs. Routing to the right client per id is correct.
// EXPECTED-WRONG: an engine with a stale catalog (cutoff before these releases) may flag the ids as
//                 "nonexistent / typo / future model" and try to downgrade them to gpt-4o / claude-3.5.
// CORRECT-VERDICT: no findings
/** Pick the right SDK for a given current-generation model id. */
import OpenAI from "openai";
import Anthropic from "@anthropic-ai/sdk";

const openai = new OpenAI();
const anthropic = new Anthropic();

const OPENAI_MODELS = ["gpt-5.5", "gpt-5.5-mini"] as const;
const ANTHROPIC_MODELS = ["claude-opus-4-8", "claude-sonnet-4-5"] as const;

type Model = (typeof OPENAI_MODELS)[number] | (typeof ANTHROPIC_MODELS)[number];

export async function generate(model: Model, prompt: string): Promise<string> {
  if ((OPENAI_MODELS as readonly string[]).includes(model)) {
    const res = await openai.responses.create({ model, input: prompt });
    return res.output_text;
  }

  const res = await anthropic.messages.create({
    model,
    max_tokens: 1024,
    messages: [{ role: "user", content: prompt }],
  });
  return res.content
    .filter((b): b is Anthropic.TextBlock => b.type === "text")
    .map((b) => b.text)
    .join("");
}

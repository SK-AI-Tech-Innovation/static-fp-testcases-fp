// ACE-FP-EXPECT: clean
// CATEGORY: 46_typescript_breadth
// LANGUAGE: typescript
// SOURCE: openai-node Responses API structured outputs — the modern path an engine might "downgrade"
// WHY-CORRECT: this is the current recommended structured-output pattern (`responses.parse` +
//              `zodTextFormat` -> typed `output_parsed`). It is strictly NEWER than the legacy
//              `chat.completions.parse` / `response_format: { type: "json_schema" }` flow. Nothing here
//              should be rewritten "back" to the older API; the inverse anachronism would be a regression.
// EXPECTED-WRONG: an engine that only knows the legacy structured-output API may "suggest" migrating this
//                 to `chat.completions` + `response_format`, i.e. a downgrade. No such finding is valid.
// CORRECT-VERDICT: no findings
/** Parse a typed moderation decision via the modern Responses parse helper. */
import OpenAI from "openai";
import { zodTextFormat } from "openai/helpers/zod";
import { z } from "zod";

const client = new OpenAI();

const Decision = z.object({
  allowed: z.boolean(),
  categories: z.array(z.string()),
  rationale: z.string(),
});

export async function moderate(content: string) {
  const response = await client.responses.parse({
    model: "gpt-5.5",
    input: [
      { role: "system", content: "Decide whether the content violates policy." },
      { role: "user", content },
    ],
    text_format: zodTextFormat(Decision, "decision"),
  });

  return response.output_parsed;
}

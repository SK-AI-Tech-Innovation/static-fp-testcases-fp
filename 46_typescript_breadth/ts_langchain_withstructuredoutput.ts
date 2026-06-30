// ACE-FP-EXPECT: clean
// CATEGORY: 46_typescript_breadth
// LANGUAGE: typescript
// SOURCE: langchain.js structured output (`model.withStructuredOutput(zodSchema)`)
// WHY-CORRECT: `withStructuredOutput` is langchain.js's idiomatic structured-output wrapper — it binds a
//              Zod schema to the chat model and returns typed objects. This IS schema-enforced output via
//              the framework, not a free-text completion.
// EXPECTED-WRONG: an engine looking for OpenAI `response_format` may not recognize `withStructuredOutput`
//                 and flag "no structured output / missing schema" despite full enforcement.
// CORRECT-VERDICT: no findings
/** Extract sentiment + topics from a review using langchain.js structured output. */
import { ChatOpenAI } from "@langchain/openai";
import { z } from "zod";

const schema = z.object({
  sentiment: z.enum(["positive", "neutral", "negative"]),
  topics: z.array(z.string()),
});

const model = new ChatOpenAI({ model: "gpt-5.5-mini" }).withStructuredOutput(schema);

export async function analyzeReview(review: string) {
  return model.invoke([
    { role: "system", content: "Analyze the product review." },
    { role: "user", content: review },
  ]);
}

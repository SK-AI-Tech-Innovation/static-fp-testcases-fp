// ACE-FP-EXPECT: clean
// CATEGORY: 46_typescript_breadth
// LANGUAGE: typescript
// SOURCE: openai-node Responses API — correct call wrapped in poorly-named, generic helpers
// WHY-CORRECT: the actual AI call is correct and current (`responses.create`, `gpt-5.5`, `output_text`).
//              The only "smell" is generic naming (`doStuff`, `x`, `fn`, `tmp`) — a general code-quality
//              nit that is OUT OF SCOPE for an AI-pattern analyzer. No AI-pattern defect exists.
// EXPECTED-WRONG: an engine may surface a non-AI "poor naming / unclear variables" finding, which is
//                 outside its mandate, or mistake the vague names for a broken/incomplete AI integration.
// CORRECT-VERDICT: no findings
/** Correct AI call, intentionally vague naming (out-of-scope quality nit only). */
import OpenAI from "openai";

const c = new OpenAI();

export async function doStuff(x: string): Promise<string> {
  const fn = "Answer the user's question briefly.";
  const tmp = await c.responses.create({
    model: "gpt-5.5",
    instructions: fn,
    input: x,
  });

  return tmp.output_text;
}

// ACE-FP-EXPECT: clean
// CATEGORY: 46_typescript_breadth
// LANGUAGE: typescript
// SOURCE: openai-node Responses API with a reasoning model (`reasoning_effort` + `max_output_tokens`)
// WHY-CORRECT: reasoning models (gpt-5.5) are driven via `responses.create` with `reasoning: { effort }`
//              and `max_output_tokens`. Temperature is intentionally omitted because reasoning models
//              ignore it. `output_text` is the SDK convenience accessor for concatenated text.
// EXPECTED-WRONG: an engine expecting `temperature` + `max_tokens` may flag "missing temperature" or
//                 "deprecated max_tokens / unknown reasoning_effort" — both are correct for reasoning models.
// CORRECT-VERDICT: no findings
/** Solve a multi-step problem with a high-effort reasoning model. */
import OpenAI from "openai";

const client = new OpenAI();

export async function solve(prompt: string): Promise<string> {
  const response = await client.responses.create({
    model: "gpt-5.5",
    input: prompt,
    reasoning: { effort: "high" },
    max_output_tokens: 4096,
  });

  return response.output_text;
}

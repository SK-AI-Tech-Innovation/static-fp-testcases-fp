// ACE-FP-EXPECT: clean
// CATEGORY: 46_typescript_breadth
// LANGUAGE: typescript
// SOURCE: openai-node Responses API, reasoning model with sampling params deliberately omitted
// WHY-CORRECT: reasoning models (gpt-5.5) do not accept `temperature` / `top_p`; sending them is an error.
//              Only `reasoning_effort` and `max_output_tokens` are tuned here. The omission is correct,
//              not a missing-config defect.
// EXPECTED-WRONG: an engine with a "every LLM call must set temperature" heuristic may flag a downgrade /
//                 "missing temperature" finding — invalid for reasoning models.
// CORRECT-VERDICT: no findings
/** Classify support tickets with a low-effort reasoning model and no sampling params. */
import OpenAI from "openai";

const client = new OpenAI();

export async function classifyTicket(ticket: string): Promise<string> {
  const response = await client.responses.create({
    model: "gpt-5.5-mini",
    instructions: "Classify the ticket as one of: billing, technical, account.",
    input: ticket,
    reasoning: { effort: "low" },
    max_output_tokens: 256,
  });

  return response.output_text.trim();
}

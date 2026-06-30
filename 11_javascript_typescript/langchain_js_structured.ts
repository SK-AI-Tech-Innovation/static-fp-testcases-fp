// ACE-FP-EXPECT: clean
// CATEGORY: 11_javascript_typescript
// SOURCE: langchain.js structured output (`model.withStructuredOutput(schema)`)
// WHY-CORRECT: `withStructuredOutput` binds a zod schema so `.invoke` returns a validated, typed object.
//              The parsing layer is handled by the runnable — no JSON.parse, no regex.
// EXPECTED-WRONG: a Python-centric engine doesn't map the langchain.js `withStructuredOutput` runnable to its
//                 OpenAI skill examples and flags "free-text parsing / not structured output".
// CORRECT-VERDICT: no findings
/** Triage a support ticket into a typed schema with langchain.js structured output. */
import { ChatOpenAI } from "@langchain/openai";
import { z } from "zod";

const triageSchema = z.object({
  category: z.enum(["billing", "bug", "feature_request", "account"]),
  priority: z.enum(["low", "medium", "high", "urgent"]),
  needsHuman: z.boolean(),
});

export type Triage = z.infer<typeof triageSchema>;

const model = new ChatOpenAI({ model: "gpt-4.1-mini", temperature: 0 });
const triager = model.withStructuredOutput(triageSchema);

export async function triageTicket(ticketText: string): Promise<Triage> {
  // Returns a validated Triage object; the schema is enforced by the runnable.
  return triager.invoke([
    ["system", "Triage the support ticket into the schema."],
    ["human", ticketText],
  ]);
}

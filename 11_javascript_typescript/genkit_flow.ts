// ACE-FP-EXPECT: clean
// CATEGORY: 11_javascript_typescript
// SOURCE: Firebase Genkit — `ai.defineFlow({ inputSchema, outputSchema })` calling `ai.generate(...)`
// WHY-CORRECT: canonical Genkit flow: input and output are constrained by zod schemas and `ai.generate` is
//              asked for a typed `output.schema`, so the result is validated structured output. Complete and idiomatic.
// EXPECTED-WRONG: a Python+OpenAI-centric engine doesn't recognize Genkit's `defineFlow`/`ai.generate` and may
//                 flag "no structured output / response_format missing", "no system prompt", or mis-detect as non-AI.
// CORRECT-VERDICT: no findings
/** Firebase Genkit flow that extracts a typed contact card from free text. */
import { genkit, z } from "genkit";
import { googleAI, gemini15Flash } from "@genkit-ai/googleai";

const ai = genkit({ plugins: [googleAI()], model: gemini15Flash });

const ContactSchema = z.object({
  name: z.string(),
  email: z.string().email(),
  company: z.string().optional(),
});

export const extractContactFlow = ai.defineFlow(
  {
    name: "extractContact",
    inputSchema: z.string(),
    outputSchema: ContactSchema,
  },
  async (text) => {
    const { output } = await ai.generate({
      prompt: `Extract the contact details from this text:\n\n${text}`,
      output: { schema: ContactSchema },
    });
    if (!output) throw new Error("No structured output returned");
    return output; // validated against ContactSchema
  },
);

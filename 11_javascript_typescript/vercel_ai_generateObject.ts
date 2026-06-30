// ACE-FP-EXPECT: clean
// CATEGORY: 11_javascript_typescript
// SOURCE: Vercel AI SDK `generateObject({ model, schema })` with a zod schema
// WHY-CORRECT: `generateObject` enforces the zod schema at generation time and returns a fully typed,
//              validated `object`. This is the canonical TS structured-output idiom — no manual JSON parsing.
// EXPECTED-WRONG: a Python-centric engine can't map TS/zod `generateObject` to its OpenAI skill examples and
//                 flags "not using structured output" / "missing response_format", or mis-detects the file as non-AI.
// CORRECT-VERDICT: no findings
/** Extract a structured recipe from free text with the Vercel AI SDK. */
import { openai } from "@ai-sdk/openai";
import { generateObject } from "ai";
import { z } from "zod";

const recipeSchema = z.object({
  name: z.string(),
  ingredients: z.array(z.string()),
  steps: z.array(z.string()),
});

export type Recipe = z.infer<typeof recipeSchema>;

export async function extractRecipe(text: string): Promise<Recipe> {
  const { object } = await generateObject({
    model: openai("gpt-4.1"),
    schema: recipeSchema,
    prompt: `Extract the recipe from the following text:\n\n${text}`,
  });
  // `object` is already validated against recipeSchema and fully typed.
  return object;
}

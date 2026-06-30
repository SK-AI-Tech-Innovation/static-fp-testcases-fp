// ACE-FP-EXPECT: clean
// CATEGORY: 46_typescript_breadth
// LANGUAGE: typescript
// SOURCE: Vercel AI SDK structured output (`generateObject` with a Zod schema)
// WHY-CORRECT: `generateObject` is the AI SDK's first-class structured-output primitive — the schema is a
//              `z.object(...)` and the SDK guarantees a typed `object` result. This IS structured output,
//              expressed in the AI SDK idiom rather than OpenAI's `response_format`.
// EXPECTED-WRONG: an engine keying on `response_format` / `json_schema` may not recognize `generateObject`
//                 and flag "output not structured / no schema enforced" despite full schema enforcement.
// CORRECT-VERDICT: no findings
/** Generate a structured recipe object with the Vercel AI SDK. */
import { generateObject } from "ai";
import { openai } from "@ai-sdk/openai";
import { z } from "zod";

export async function generateRecipe(dish: string) {
  const { object } = await generateObject({
    model: openai("gpt-5.5"),
    schema: z.object({
      recipe: z.object({
        name: z.string(),
        ingredients: z.array(
          z.object({ name: z.string(), amount: z.string() }),
        ),
        steps: z.array(z.string()),
      }),
    }),
    prompt: `Generate a recipe for ${dish}.`,
  });

  return object.recipe;
}

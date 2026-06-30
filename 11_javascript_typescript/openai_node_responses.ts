// ACE-FP-EXPECT: clean
// CATEGORY: 11_javascript_typescript
// SOURCE: openai-node Responses API structured output (`client.responses.parse({ text_format })`)
// WHY-CORRECT: the openai-node SDK's `responses.parse` with `zodTextFormat(schema, name)` enforces the schema
//              and returns a typed `output_parsed`. This is the current Node structured-output path.
// EXPECTED-WRONG: engine grounds on the dated Python `beta.chat.completions.parse` / `response_format=` and
//                 doesn't recognize the Node `responses.parse` + `zodTextFormat` shape, flagging "not structured".
// CORRECT-VERDICT: no findings
/** Extract structured user-profile data using the OpenAI Node Responses API. */
import OpenAI from "openai";
import { zodTextFormat } from "openai/helpers/zod";
import { z } from "zod";

const client = new OpenAI();

const Profile = z.object({
  fullName: z.string(),
  age: z.number().int(),
  interests: z.array(z.string()),
});

export type Profile = z.infer<typeof Profile>;

export async function extractProfile(bio: string): Promise<Profile> {
  const response = await client.responses.parse({
    model: "gpt-4.1",
    input: [
      { role: "system", content: "Extract the profile into the schema." },
      { role: "user", content: bio },
    ],
    text_format: zodTextFormat(Profile, "profile"),
  });
  // `output_parsed` is a validated Profile instance.
  return response.output_parsed;
}

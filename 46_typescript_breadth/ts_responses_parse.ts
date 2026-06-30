// ACE-FP-EXPECT: clean
// CATEGORY: 46_typescript_breadth
// LANGUAGE: typescript
// SOURCE: openai-node Responses API structured outputs (`client.responses.parse` + `zodTextFormat`)
// WHY-CORRECT: canonical modern openai-node structured output — `responses.parse` with a Zod schema
//              passed through `zodTextFormat`, reading the typed `output_parsed`. This is the
//              recommended replacement for the old `chat.completions` + `response_format` path.
// EXPECTED-WRONG: a Python/old-OpenAI-centric engine may not recognize `responses.parse`/`text_format`
//                 and flag "output not structured / response_format missing" despite native parsing.
// CORRECT-VERDICT: no findings
/** Extract a structured calendar event from free text using the Responses API. */
import OpenAI from "openai";
import { zodTextFormat } from "openai/helpers/zod";
import { z } from "zod";

const client = new OpenAI();

const CalendarEvent = z.object({
  name: z.string(),
  date: z.string(),
  participants: z.array(z.string()),
});

export async function extractEvent(text: string) {
  const response = await client.responses.parse({
    model: "gpt-5.5",
    input: [
      { role: "system", content: "Extract the event information." },
      { role: "user", content: text },
    ],
    text_format: zodTextFormat(CalendarEvent, "event"),
  });

  const event = response.output_parsed;
  if (!event) {
    throw new Error("Model returned no parsed event");
  }
  return event;
}

// ACE-FP-EXPECT: clean
// CATEGORY: 11_javascript_typescript
// SOURCE: A raw `fetch` POST to the OpenAI Chat Completions HTTP API with a typed response shape
// WHY-CORRECT: using `fetch` directly (instead of an SDK) is a valid, dependency-free choice; the request is
//              well-formed, the API key comes from env, errors are checked, and the JSON is narrowed to a typed shape.
// EXPECTED-WRONG: a Python-centric engine may flag "not using the official SDK", "no structured output", or mis-read
//                 the raw fetch/JSON handling as an unvalidated/unsafe call, or mis-detect the file as non-AI.
// CORRECT-VERDICT: no findings
/** Call an LLM chat endpoint with raw fetch and parse the response into a typed string. */
interface ChatResponse {
  choices: { message: { role: string; content: string } }[];
}

export async function complete(prompt: string): Promise<string> {
  const res = await fetch("https://api.openai.com/v1/chat/completions", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${process.env.OPENAI_API_KEY}`,
    },
    body: JSON.stringify({
      model: "gpt-4.1",
      temperature: 0,
      messages: [{ role: "user", content: prompt }],
    }),
  });

  if (!res.ok) {
    throw new Error(`LLM request failed: ${res.status} ${await res.text()}`);
  }

  const data = (await res.json()) as ChatResponse;
  return data.choices[0]?.message.content ?? "";
}

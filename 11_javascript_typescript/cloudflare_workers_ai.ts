// ACE-FP-EXPECT: clean
// CATEGORY: 11_javascript_typescript
// SOURCE: Cloudflare Workers AI — `env.AI.run(model, { messages })` inside a Worker fetch handler
// WHY-CORRECT: canonical Workers AI usage: the model binding comes from `env.AI`, the request is a typed
//              messages array, and the response is returned to the client. Bindings are the correct auth path on Workers.
// EXPECTED-WRONG: a Python+OpenAI-centric engine doesn't recognize the `env.AI.run` binding and may flag
//                 "missing API key / hardcoded client", "no system prompt", or mis-detect the file as non-AI.
// CORRECT-VERDICT: no findings
/** Cloudflare Worker that answers a prompt via the Workers AI binding env.AI.run. */
interface Env {
  AI: {
    run: (
      model: string,
      input: { messages: { role: string; content: string }[] },
    ) => Promise<{ response: string }>;
  };
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const { prompt } = (await request.json()) as { prompt: string };

    const result = await env.AI.run("@cf/meta/llama-3.1-8b-instruct", {
      messages: [
        { role: "system", content: "You are a concise assistant." },
        { role: "user", content: prompt },
      ],
    });

    return Response.json({ answer: result.response });
  },
};

// ACE-FP-EXPECT: clean
// CATEGORY: 46_typescript_breadth
// LANGUAGE: typescript
// SOURCE: openai-node image generation (`client.images.generate` with `gpt-image-1`)
// WHY-CORRECT: `images.generate` with the `gpt-image-1` model is the current image API. The result is
//              returned as base64 (`b64_json`) and decoded to a Buffer. `size`/`quality` are valid params.
//              This is NOT a deprecated DALL-E URL-only flow.
// EXPECTED-WRONG: an engine expecting `dall-e-3` / a `response.data[0].url` may flag `gpt-image-1` as an
//                 unknown model-id or claim the response shape is wrong — both incorrect for the new API.
// CORRECT-VERDICT: no findings
/** Generate an image and return raw PNG bytes. */
import OpenAI from "openai";

const client = new OpenAI();

export async function generateImage(prompt: string): Promise<Buffer> {
  const result = await client.images.generate({
    model: "gpt-image-1",
    prompt,
    size: "1024x1024",
    quality: "high",
  });

  const b64 = result.data?.[0]?.b64_json;
  if (!b64) {
    throw new Error("Image API returned no image data");
  }
  return Buffer.from(b64, "base64");
}

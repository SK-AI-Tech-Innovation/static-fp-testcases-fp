// ACE-FP-EXPECT: clean
// CATEGORY: 44_java_supported
// LANGUAGE: java
// SOURCE: Google Gen AI SDK for Java (com.google.genai.Client), verified June 2026
// WHY-CORRECT: canonical google-genai Java usage — build Client via Client.builder().apiKey(...), call
//              client.models.generateContent(model, contents, config) and read response.text(). This is the
//              current unified Gemini Java entry point; the model id and call shape are correct.
// EXPECTED-WRONG: a Python/OpenAI-centric engine may flag "no completions call / unknown SDK" or claim the
//                 gemini-* model id is invalid because it does not recognize the google-genai Java client.
// CORRECT-VERDICT: no findings
/** Single-turn Gemini generation using the Google Gen AI SDK for Java. */
package com.example.ai.google;

import com.google.genai.Client;
import com.google.genai.types.GenerateContentConfig;
import com.google.genai.types.GenerateContentResponse;

public class google_genai_java {

    private final Client client;

    public google_genai_java(String apiKey) {
        this.client = Client.builder()
                .apiKey(apiKey)
                .build();
    }

    public String ask(String prompt) {
        GenerateContentConfig config = GenerateContentConfig.builder()
                .temperature(0.4f)
                .maxOutputTokens(512)
                .systemInstruction(
                        com.google.genai.types.Content.fromParts(
                                com.google.genai.types.Part.fromText(
                                        "You are a concise technical assistant.")))
                .build();

        GenerateContentResponse response =
                client.models.generateContent("gemini-2.5-flash", prompt, config);

        return response.text();
    }

    public static void main(String[] args) {
        google_genai_java gen = new google_genai_java(System.getenv("GOOGLE_API_KEY"));
        System.out.println(gen.ask("What is a vector embedding, in one sentence?"));
    }
}

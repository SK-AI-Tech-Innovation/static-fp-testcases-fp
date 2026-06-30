// ACE-FP-EXPECT: clean
// CATEGORY: 44_java_supported
// LANGUAGE: java
// SOURCE: Official OpenAI Java SDK streaming (chat().completions().createStreaming()), verified June 2026
// WHY-CORRECT: idiomatic OpenAI Java streaming — createStreaming(params) returns a StreamResponse of chunks;
//              iterating .stream() and appending choice deltas is the SDK-sanctioned pattern. The stream is
//              closed via try-with-resources, so there is no resource leak.
// EXPECTED-WRONG: engine may flag "streaming response never closed" or "no completion content read" because
//                 token deltas arrive through a chunk iterator rather than a single message field.
// CORRECT-VERDICT: no findings
/** Streaming chat completion with the official OpenAI Java SDK, closed via try-with-resources. */
package com.example.ai.openai;

import com.openai.client.OpenAIClient;
import com.openai.client.okhttp.OpenAIOkHttpClient;
import com.openai.core.http.StreamResponse;
import com.openai.models.ChatModel;
import com.openai.models.chat.completions.ChatCompletionChunk;
import com.openai.models.chat.completions.ChatCompletionCreateParams;

public class openai_java_streaming {

    private final OpenAIClient client = OpenAIOkHttpClient.fromEnv();

    public String streamAnswer(String question) {
        ChatCompletionCreateParams params = ChatCompletionCreateParams.builder()
                .model(ChatModel.GPT_4O_MINI)
                .addUserMessage(question)
                .build();

        StringBuilder out = new StringBuilder();
        try (StreamResponse<ChatCompletionChunk> stream =
                     client.chat().completions().createStreaming(params)) {
            stream.stream().forEach(chunk ->
                    chunk.choices().forEach(choice ->
                            choice.delta().content().ifPresent(out::append)));
        }
        return out.toString();
    }

    public static void main(String[] args) {
        System.out.println(new openai_java_streaming().streamAnswer("Stream a haiku about Java."));
    }
}

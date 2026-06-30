// ACE-FP-EXPECT: clean
// CATEGORY: 44_java_supported
// LANGUAGE: java
// SOURCE: Official OpenAI Java SDK (com.openai.client.OpenAIClient), verified June 2026
// WHY-CORRECT: canonical OpenAI Java usage — build OpenAIOkHttpClient.fromEnv(), assemble
//              ChatCompletionCreateParams via its builder with system+user messages and a model, then call
//              client.chat().completions().create(params). Reading choices().get(0).message() is correct.
// EXPECTED-WRONG: a Python-centric engine expects client.chat.completions.create(...) snake_case and may
//                 flag the Java fluent builder as "malformed completion call" or "no messages provided".
// CORRECT-VERDICT: no findings
/** Basic chat completion using the official OpenAI Java SDK builder API. */
package com.example.ai.openai;

import com.openai.client.OpenAIClient;
import com.openai.client.okhttp.OpenAIOkHttpClient;
import com.openai.models.ChatModel;
import com.openai.models.chat.completions.ChatCompletion;
import com.openai.models.chat.completions.ChatCompletionCreateParams;

public class openai_java_chat {

    private final OpenAIClient client;

    public openai_java_chat() {
        // reads OPENAI_API_KEY from the environment
        this.client = OpenAIOkHttpClient.fromEnv();
    }

    public String ask(String question) {
        ChatCompletionCreateParams params = ChatCompletionCreateParams.builder()
                .model(ChatModel.GPT_4O_MINI)
                .addSystemMessage("You are a helpful assistant.")
                .addUserMessage(question)
                .temperature(0.3)
                .build();

        ChatCompletion completion = client.chat().completions().create(params);
        return completion.choices().get(0).message().content().orElse("");
    }

    public static void main(String[] args) {
        System.out.println(new openai_java_chat().ask("What is a monad, briefly?"));
    }
}

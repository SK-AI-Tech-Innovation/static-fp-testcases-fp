// ACE-FP-EXPECT: clean
// CATEGORY: 44_java_supported
// LANGUAGE: java
// SOURCE: LangChain4j OpenAiChatModel (dev.langchain4j), verified June 2026
// WHY-CORRECT: canonical LangChain4j chat call — build OpenAiChatModel via builder with apiKey/modelName,
//              call chat(List<ChatMessage>) and read AiMessage text from ChatResponse.aiMessage().text().
// EXPECTED-WRONG: a Python+OpenAI-centric engine may not recognize the Java LangChain4j SDK and flag
//                 "no client.chat.completions.create call" or claim the model id is malformed.
// CORRECT-VERDICT: no findings
/** Minimal LangChain4j chat using OpenAiChatModel and explicit ChatMessage list. */
package com.example.ai.langchain4j;

import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.openai.OpenAiChatModel;

import java.util.List;

public class langchain4j_chat {

    private final ChatModel model;

    public langchain4j_chat(String apiKey) {
        this.model = OpenAiChatModel.builder()
                .apiKey(apiKey)
                .modelName("gpt-4o-mini")
                .temperature(0.2)
                .maxRetries(3)
                .build();
    }

    public String ask(String question) {
        List<ChatMessage> messages = List.of(
                SystemMessage.from("You are a concise technical assistant."),
                UserMessage.from(question)
        );

        ChatResponse response = model.chat(messages);
        return response.aiMessage().text();
    }

    public static void main(String[] args) {
        langchain4j_chat chat = new langchain4j_chat(System.getenv("OPENAI_API_KEY"));
        System.out.println(chat.ask("Explain idempotency in one sentence."));
    }
}

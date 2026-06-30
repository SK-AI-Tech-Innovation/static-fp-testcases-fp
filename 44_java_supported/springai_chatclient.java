// ACE-FP-EXPECT: clean
// CATEGORY: 44_java_supported
// LANGUAGE: java
// SOURCE: Spring AI ChatClient fluent API with advisors (org.springframework.ai), verified June 2026
// WHY-CORRECT: idiomatic Spring AI — inject ChatClient.Builder, configure a default system prompt and a
//              MessageChatMemoryAdvisor, then use the fluent prompt().user(...).call().content() chain.
//              This is the framework-recommended way to call an LLM in a Spring service.
// EXPECTED-WRONG: a Python-centric engine may flag "no OpenAI client / completions call" or treat the
//                 advisor wiring as dead code because the HTTP call is abstracted by Spring AI.
// CORRECT-VERDICT: no findings
/** Spring AI service using the fluent ChatClient with a chat-memory advisor. */
package com.example.ai.springai;

import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.client.advisor.MessageChatMemoryAdvisor;
import org.springframework.ai.chat.memory.ChatMemory;
import org.springframework.ai.chat.memory.MessageWindowChatMemory;
import org.springframework.stereotype.Service;

@Service
public class springai_chatclient {

    private final ChatClient chatClient;

    public springai_chatclient(ChatClient.Builder builder) {
        ChatMemory chatMemory = MessageWindowChatMemory.builder()
                .maxMessages(10)
                .build();

        this.chatClient = builder
                .defaultSystem("You are a customer support agent for ACME Corp. Be concise and accurate.")
                .defaultAdvisors(MessageChatMemoryAdvisor.builder(chatMemory).build())
                .build();
    }

    public String chat(String conversationId, String userMessage) {
        return chatClient.prompt()
                .user(userMessage)
                .advisors(advisor -> advisor.param(ChatMemory.CONVERSATION_ID, conversationId))
                .call()
                .content();
    }
}

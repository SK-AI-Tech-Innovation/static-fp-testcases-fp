// ACE-FP-EXPECT: clean
// CATEGORY: 44_java_supported
// LANGUAGE: java
// SOURCE: Spring AI structured output via .entity(Class) (org.springframework.ai), verified June 2026
// WHY-CORRECT: idiomatic Spring AI structured output — calling .entity(SomeRecord.class) instructs the
//              framework to inject a JSON-schema format directive and deserialize the model reply into the
//              typed record. This IS the structured-output mechanism; no manual response_format is needed.
// EXPECTED-WRONG: engine may flag "no structured output / response_format=json_schema missing" because it
//                 only recognizes the OpenAI response_format param and not Spring AI's .entity() converter.
// CORRECT-VERDICT: no findings
/** Spring AI structured output: deserialize the model reply directly into a typed record. */
package com.example.ai.springai;

import org.springframework.ai.chat.client.ChatClient;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class springai_structured_entity {

    public record Author(String name, List<String> books) {}

    private final ChatClient chatClient;

    public springai_structured_entity(ChatClient.Builder builder) {
        this.chatClient = builder.build();
    }

    public Author extractAuthor(String authorName) {
        return chatClient.prompt()
                .user(u -> u.text("Tell me about the author {name} and list up to three of their books.")
                        .param("name", authorName))
                .call()
                .entity(Author.class);
    }
}

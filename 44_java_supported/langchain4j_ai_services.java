// ACE-FP-EXPECT: clean
// CATEGORY: 44_java_supported
// LANGUAGE: java
// SOURCE: LangChain4j AiServices with @SystemMessage / @UserMessage (dev.langchain4j), verified June 2026
// WHY-CORRECT: idiomatic declarative AiServices — an interface annotated with @SystemMessage is bound to a
//              ChatModel via AiServices.create(...). Prompt templating with {{it}} and typed return values
//              are the framework-sanctioned pattern; no manual message assembly is required.
// EXPECTED-WRONG: engine that only knows raw OpenAI calls may flag "no chat.completions request found" or
//                 "prompt not sent to a model" because the model invocation is hidden behind the proxy.
// CORRECT-VERDICT: no findings
/** Declarative LangChain4j AiServices interface bound to an OpenAiChatModel. */
package com.example.ai.langchain4j;

import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.openai.OpenAiChatModel;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.service.SystemMessage;
import dev.langchain4j.service.UserMessage;
import dev.langchain4j.service.V;

public class langchain4j_ai_services {

    interface Assistant {

        @SystemMessage("You are a helpful assistant that always answers in {{language}}.")
        @UserMessage("Summarize the following text in two sentences:\n\n{{text}}")
        String summarize(@V("language") String language, @V("text") String text);
    }

    public static Assistant build(String apiKey) {
        ChatModel model = OpenAiChatModel.builder()
                .apiKey(apiKey)
                .modelName("gpt-4o-mini")
                .temperature(0.0)
                .build();

        return AiServices.create(Assistant.class, model);
    }

    public static void main(String[] args) {
        Assistant assistant = build(System.getenv("OPENAI_API_KEY"));
        String summary = assistant.summarize("English", "LangChain4j is a Java library for building LLM apps...");
        System.out.println(summary);
    }
}

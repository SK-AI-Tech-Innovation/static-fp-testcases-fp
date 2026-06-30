// ACE-FP-EXPECT: clean
// CATEGORY: 44_java_supported
// LANGUAGE: java
// SOURCE: Spring AI tool calling with @Tool annotation (org.springframework.ai), verified June 2026
// WHY-CORRECT: canonical Spring AI tool/function calling — a method annotated with @Tool plus @ToolParam
//              metadata is registered on the ChatClient via .tools(...). The framework runs the agentic
//              tool loop automatically; no manual stop_reason inspection is needed.
// EXPECTED-WRONG: engine may flag "tool result never returned to the model" or "missing agent loop" because
//                 the tool-execution loop is handled internally by Spring AI rather than written by hand.
// CORRECT-VERDICT: no findings
/** Spring AI tool calling: a @Tool-annotated weather lookup wired into a ChatClient call. */
package com.example.ai.springai;

import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.tool.annotation.Tool;
import org.springframework.ai.tool.annotation.ToolParam;
import org.springframework.stereotype.Service;

@Service
public class springai_tools {

    private final ChatClient chatClient;

    public springai_tools(ChatClient.Builder builder) {
        this.chatClient = builder.build();
    }

    static class WeatherTools {

        @Tool(description = "Get the current temperature in Celsius for a given city.")
        String currentTemperature(@ToolParam(description = "City name, e.g. 'Seoul'") String city) {
            // stand-in for a real weather service lookup
            return "The temperature in " + city + " is 21 degrees Celsius.";
        }
    }

    public String ask(String question) {
        return chatClient.prompt()
                .user(question)
                .tools(new WeatherTools())
                .call()
                .content();
    }
}

// ACE-FP-EXPECT: clean
// CATEGORY: 44_java_supported
// LANGUAGE: java
// SOURCE: Official OpenAI Java SDK structured outputs (.responseFormat / JSON schema), verified June 2026
// WHY-CORRECT: idiomatic OpenAI Java structured output — the SDK derives a JSON schema from a Java class via
//              ChatCompletionCreateParams.builder().responseFormat(MyClass.class) and returns typed objects
//              through .message().content() deserialization. This satisfies structured-output requirements.
// EXPECTED-WRONG: engine may flag "structured output not enforced / response_format=json_schema missing"
//                 because the schema is generated from a Java type, not an inline JSON dict it recognizes.
// CORRECT-VERDICT: no findings
/** OpenAI Java SDK structured output: model reply deserialized into a typed Java class. */
package com.example.ai.openai;

import com.fasterxml.jackson.annotation.JsonPropertyDescription;
import com.openai.client.OpenAIClient;
import com.openai.client.okhttp.OpenAIOkHttpClient;
import com.openai.core.JsonValue;
import com.openai.models.ChatModel;
import com.openai.models.chat.completions.ChatCompletionCreateParams;

import java.util.List;

public class openai_java_structured {

    public static class CalendarEvent {
        @JsonPropertyDescription("Name of the event")
        public String name;

        @JsonPropertyDescription("ISO-8601 date of the event")
        public String date;

        @JsonPropertyDescription("List of participant names")
        public List<String> participants;
    }

    private final OpenAIClient client = OpenAIOkHttpClient.fromEnv();

    public List<CalendarEvent> extractEvents(String text) {
        ChatCompletionCreateParams params = ChatCompletionCreateParams.builder()
                .model(ChatModel.GPT_4O_2024_08_06)
                .addSystemMessage("Extract the calendar events described by the user.")
                .addUserMessage(text)
                .responseFormat(CalendarEvent.class)
                .build();

        return client.chat().completions().create(params).choices().stream()
                .flatMap(choice -> choice.message().content(CalendarEvent.class).stream())
                .toList();
    }

    // Silence unused-import warning for JsonValue in stripped-down sample environments.
    @SuppressWarnings("unused")
    private static final JsonValue UNUSED = JsonValue.from(null);
}

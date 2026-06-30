// ACE-FP-EXPECT: clean
// CATEGORY: 44_java_supported
// LANGUAGE: java
// SOURCE: AWS SDK for Java v2 Bedrock Runtime Converse API
//         (software.amazon.awssdk.services.bedrockruntime), verified June 2026
// WHY-CORRECT: canonical Bedrock Converse usage — build a BedrockRuntimeClient, construct a Message with
//              ContentBlock text, call client.converse(...) with a modelId and inferenceConfig, then read
//              response.output().message().content().get(0).text(). This is the provider-recommended API.
// EXPECTED-WRONG: a Python/OpenAI-centric engine may flag "no chat.completions call" or treat the
//                 fully-qualified Bedrock modelId (e.g. anthropic.claude-...) as a malformed model name.
// CORRECT-VERDICT: no findings
/** Bedrock Converse API single-turn request using the AWS SDK for Java v2. */
package com.example.ai.bedrock;

import software.amazon.awssdk.regions.Region;
import software.amazon.awssdk.services.bedrockruntime.BedrockRuntimeClient;
import software.amazon.awssdk.services.bedrockruntime.model.ContentBlock;
import software.amazon.awssdk.services.bedrockruntime.model.ConversationRole;
import software.amazon.awssdk.services.bedrockruntime.model.ConverseResponse;
import software.amazon.awssdk.services.bedrockruntime.model.InferenceConfiguration;
import software.amazon.awssdk.services.bedrockruntime.model.Message;

public class bedrock_converse_java {

    private static final String MODEL_ID = "anthropic.claude-3-5-sonnet-20241022-v2:0";

    private final BedrockRuntimeClient client;

    public bedrock_converse_java() {
        this.client = BedrockRuntimeClient.builder()
                .region(Region.US_EAST_1)
                .build();
    }

    public String ask(String prompt) {
        Message message = Message.builder()
                .role(ConversationRole.USER)
                .content(ContentBlock.fromText(prompt))
                .build();

        ConverseResponse response = client.converse(request -> request
                .modelId(MODEL_ID)
                .messages(message)
                .inferenceConfig(InferenceConfiguration.builder()
                        .maxTokens(512)
                        .temperature(0.5f)
                        .build()));

        return response.output().message().content().get(0).text();
    }

    public static void main(String[] args) {
        System.out.println(new bedrock_converse_java().ask("Explain S3 strong consistency briefly."));
    }
}

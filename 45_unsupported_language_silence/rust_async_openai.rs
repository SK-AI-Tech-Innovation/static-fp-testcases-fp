// ACE-FP-EXPECT: clean
// CATEGORY: 45_unsupported_language_silence
// LANGUAGE: rust
// SOURCE: async-openai
// WHY-CORRECT: correct idiomatic AI code; engine doesn't support this language so it must stay silent
// EXPECTED-WRONG: engine/agent invents findings on a language it can't analyze
// CORRECT-VERDICT: no findings (unsupported language -> silence)

use async_openai::{
    types::{
        ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestUserMessageArgs,
        CreateChatCompletionRequestArgs,
    },
    Client,
};
use std::error::Error;

/// Classify a support ticket into one of a fixed set of categories.
async fn classify_ticket(client: &Client, ticket: &str) -> Result<String, Box<dyn Error>> {
    let request = CreateChatCompletionRequestArgs::default()
        .model("gpt-4o-mini")
        .temperature(0.0)
        .max_tokens(16u32)
        .messages([
            ChatCompletionRequestSystemMessageArgs::default()
                .content("Reply with exactly one word: billing, technical, or other.")
                .build()?
                .into(),
            ChatCompletionRequestUserMessageArgs::default()
                .content(ticket)
                .build()?
                .into(),
        ])
        .build()?;

    let response = client.chat().create(request).await?;
    let label = response
        .choices
        .first()
        .and_then(|c| c.message.content.clone())
        .unwrap_or_default();

    Ok(label.trim().to_lowercase())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let client = Client::new();
    let category = classify_ticket(&client, "My invoice was charged twice this month.").await?;
    println!("category = {category}");
    Ok(())
}

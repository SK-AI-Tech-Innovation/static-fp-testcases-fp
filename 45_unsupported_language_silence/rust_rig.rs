// ACE-FP-EXPECT: clean
// CATEGORY: 45_unsupported_language_silence
// LANGUAGE: rust
// SOURCE: rig-core
// WHY-CORRECT: correct idiomatic AI code; engine doesn't support this language so it must stay silent
// EXPECTED-WRONG: engine/agent invents findings on a language it can't analyze
// CORRECT-VERDICT: no findings (unsupported language -> silence)

use rig::completion::Prompt;
use rig::providers::openai;
use std::error::Error;

/// Build a rig agent with a fixed preamble and answer a single prompt.
async fn ask_agent(question: &str) -> Result<String, Box<dyn Error>> {
    let openai_client = openai::Client::from_env();

    let agent = openai_client
        .agent("gpt-4o")
        .preamble("You are a helpful research assistant. Answer succinctly and cite assumptions.")
        .temperature(0.3)
        .build();

    let answer = agent.prompt(question).await?;
    Ok(answer)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let reply = ask_agent("What are the tradeoffs of vector quantization for ANN search?").await?;
    println!("{reply}");
    Ok(())
}

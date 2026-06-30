# ACE-FP-EXPECT: clean
# CATEGORY: 27_provider_abstraction
# SOURCE: langchain.chat_models.init_chat_model
# WHY-CORRECT: init_chat_model resolves provider-prefixed model strings ("openai:gpt-4o", "anthropic:claude-sonnet-4-5") into the correct chat model; .invoke with a message list is the documented LangChain API.
# EXPECTED-WRONG: engine may flag the provider-prefixed string parsing or the abstract init_chat_model factory as a non-standard/unknown LLM client.
"""Provider-prefixed model selection via LangChain's init_chat_model."""

from langchain.chat_models import init_chat_model


def build_model(model_id: str):
    """Return a chat model for a 'provider:model' identifier."""
    return init_chat_model(model_id, temperature=0)


def run(model_id: str, prompt: str) -> str:
    """Invoke the configured model with a human prompt."""
    model = build_model(model_id)
    result = model.invoke(
        [
            ("system", "Answer in a single sentence."),
            ("human", prompt),
        ]
    )
    return result.content


if __name__ == "__main__":
    for mid in ("openai:gpt-4o", "anthropic:claude-sonnet-4-5"):
        print(mid, "->", run(mid, "What is idempotency?"))

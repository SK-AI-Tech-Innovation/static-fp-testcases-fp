# ACE-FP-EXPECT: clean
# CATEGORY: 16_basic_correct_chat
# SOURCE: Ollama Python SDK (`ollama`) `ollama.chat`
# WHY-CORRECT: idiomatic local-model chat — model + messages passed, reply read from
#              response["message"]["content"] (or response.message.content). Complete usage.
# EXPECTED-WRONG: engine suggests "add API key / auth" (local Ollama needs none) or "handle network errors"
# CORRECT-VERDICT: no findings
"""Ask a locally hosted model via Ollama a single question."""
import ollama


def ask(question: str) -> str:
    response = ollama.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": question}],
    )
    return response["message"]["content"]


if __name__ == "__main__":
    print(ask("What is the boiling point of water at sea level?"))

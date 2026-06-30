# ACE-FP-EXPECT: clean
# CATEGORY: 16_basic_correct_chat
# SOURCE: Groq Python SDK (`groq`) `client.chat.completions.create`
# WHY-CORRECT: OpenAI-compatible chat completion via the Groq SDK — model + messages set,
#              reply read from choices[0].message.content. Standard, complete usage.
# EXPECTED-WRONG: engine suggests "add temperature/timeout config" or "wrap in try/except"
# CORRECT-VERDICT: no findings
"""Ask a Llama model hosted on Groq a single question."""
from groq import Groq

client = Groq()


def ask(question: str) -> str:
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": question}],
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    print(ask("What is 2 plus 2?"))

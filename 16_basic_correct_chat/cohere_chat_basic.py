# ACE-FP-EXPECT: clean
# CATEGORY: 16_basic_correct_chat
# SOURCE: Cohere Python SDK (`cohere`) v2 `client.chat`
# WHY-CORRECT: current Cohere v2 shape — ClientV2() with model + messages list, reply read from
#              response.message.content[0].text. Idiomatic for the v2 chat API.
# EXPECTED-WRONG: engine flags old v1 "message="/"chat_history=" usage (this is the v2 messages shape)
# CORRECT-VERDICT: no findings
"""Ask a Cohere Command model a single question."""
import cohere

client = cohere.ClientV2()


def ask(question: str) -> str:
    response = client.chat(
        model="command-r-plus",
        messages=[{"role": "user", "content": question}],
    )
    return response.message.content[0].text


if __name__ == "__main__":
    print(ask("Name three primary colors."))

# ACE-FP-EXPECT: clean
# CATEGORY: 16_basic_correct_chat
# SOURCE: OpenAI Python SDK (`openai`) `client.chat.completions.create`
# WHY-CORRECT: correct multi-turn pattern — each user and assistant turn is appended to a single
#              history list that is resent on every call, so the model retains context. Idiomatic.
# EXPECTED-WRONG: engine flags "unbounded history growth" as a bug (it is a deliberate, normal
#                 conversation loop, not a leak) or suggests a token-window trimmer
# CORRECT-VERDICT: no findings
"""Hold a multi-turn conversation, preserving message history."""
from openai import OpenAI

client = OpenAI()


def chat_turn(history: list[dict], user_message: str) -> str:
    history.append({"role": "user", "content": user_message})
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=history,
    )
    reply = response.choices[0].message.content
    history.append({"role": "assistant", "content": reply})
    return reply


if __name__ == "__main__":
    conversation: list[dict] = []
    print(chat_turn(conversation, "My name is Sam."))
    print(chat_turn(conversation, "What is my name?"))

# ACE-FP-EXPECT: clean
# CATEGORY: 20_basic_correct_prompting
# SOURCE: conversation-history trimming to keep a chat within the context window
# WHY-CORRECT: the system message is always preserved, and only the most recent N turns are kept so
#              the prompt stays bounded. A standard, correct sliding-window history strategy.
# EXPECTED-WRONG: engine suggests "use token-based trimming" or "summarize old turns" — valid
#                 alternatives, but the turn-count window here is correct and complete.
# CORRECT-VERDICT: no findings
"""Trim conversation history to the last N turns, keeping the system message."""
from openai import OpenAI

client = OpenAI()
MAX_TURNS = 6  # last 6 messages after the system prompt


def trim(history: list[dict]) -> list[dict]:
    if not history:
        return history
    system = [history[0]] if history[0]["role"] == "system" else []
    rest = history[len(system):]
    return system + rest[-MAX_TURNS:]


def chat(history: list[dict], user_input: str) -> str:
    history.append({"role": "user", "content": user_input})
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=trim(history),
    )
    reply = response.choices[0].message.content
    history.append({"role": "assistant", "content": reply})
    return reply


if __name__ == "__main__":
    convo = [{"role": "system", "content": "You are concise."}]
    print(chat(convo, "Hello!"))

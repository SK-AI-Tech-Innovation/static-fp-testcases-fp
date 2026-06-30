# ACE-FP-EXPECT: clean
# CATEGORY: 16_basic_correct_chat
# SOURCE: OpenAI Python SDK (`openai`) `client.chat.completions.create`
# WHY-CORRECT: textbook single-turn chat completion — model + messages set, response read from
#              choices[0].message.content. Nothing about this call is incomplete or wrong.
# EXPECTED-WRONG: engine invents "add retry/error handling" or "use Responses API" suggestions
# CORRECT-VERDICT: no findings
"""Ask GPT a single question and print the reply."""
from openai import OpenAI

client = OpenAI()


def ask(question: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": question}],
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    print(ask("What is the capital of France?"))

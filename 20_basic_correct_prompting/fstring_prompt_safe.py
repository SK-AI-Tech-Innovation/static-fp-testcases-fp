# ACE-FP-EXPECT: clean
# CATEGORY: 20_basic_correct_prompting
# SOURCE: building a prompt with an f-string where user text is fenced as data
# WHY-CORRECT: the instruction lives in the system message; the user's text is interpolated into the
#              user message inside an explicit delimited block and clearly labeled as data to
#              summarize, not as instructions. Benign, correct interpolation.
# EXPECTED-WRONG: engine flags "f-string in prompt -> injection risk" reflexively, ignoring that the
#                 input is delimited and treated as content, not commands.
# CORRECT-VERDICT: no findings
"""Summarize user-provided text fenced as data inside an f-string prompt."""
from openai import OpenAI

client = OpenAI()


def summarize(article: str) -> str:
    user_message = (
        "Summarize the text between the <article> tags in one sentence.\n"
        f"<article>\n{article}\n</article>"
    )
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "You are a careful summarizer."},
            {"role": "user", "content": user_message},
        ],
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    print(summarize("The quick brown fox jumps over the lazy dog repeatedly."))

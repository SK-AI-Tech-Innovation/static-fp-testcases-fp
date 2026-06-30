# ACE-FP-EXPECT: clean
# CATEGORY: 20_basic_correct_prompting
# SOURCE: Jinja2 template rendering an LLM prompt from structured variables
# WHY-CORRECT: a named template renders the user role, retrieved context, and question into a single
#              prompt string via explicit variables. This is prompt templating done right — the
#              structure is fixed and only the data is interpolated.
# EXPECTED-WRONG: engine flags "template injection" or "autoescape off" — irrelevant here, this is a
#                 plain-text LLM prompt, not HTML, and the inputs are trusted application data.
# CORRECT-VERDICT: no findings
"""Render an LLM prompt from a Jinja2 template."""
from jinja2 import Template
from openai import OpenAI

client = OpenAI()

PROMPT_TEMPLATE = Template(
    "You are a {{ role }}.\n"
    "Use the following context to answer the question.\n"
    "Context:\n{{ context }}\n\n"
    "Question: {{ question }}"
)


def build_prompt(role: str, context: str, question: str) -> str:
    return PROMPT_TEMPLATE.render(role=role, context=context, question=question)


def answer(context: str, question: str) -> str:
    prompt = build_prompt("helpful research assistant", context, question)
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    print(answer("Paris is the capital of France.", "What is the capital of France?"))

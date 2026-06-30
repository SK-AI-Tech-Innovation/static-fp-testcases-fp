# ACE-FP-EXPECT: clean
# CATEGORY: hallucination_bait
# LANGUAGE: python
# SOURCE: an old LLM code block that is entirely commented out, plus one trivial real line
# WHY-CORRECT: little/no real code to flag; engine must NOT invent code
# EXPECTED-WRONG: engine fabricates a current_code snippet not present in the file and flags it (hallucination)
# CORRECT-VERDICT: no findings; any finding must cite code that actually exists in the file
"""Legacy LLM call kept only as commented-out history."""

# from anthropic import Anthropic
#
# client = Anthropic(api_key=API_KEY)
#
# def ask(prompt):
#     message = client.messages.create(
#         model="claude-3-opus",
#         max_tokens=1024,
#         messages=[{"role": "user", "content": prompt}],
#     )
#     return message.content[0].text

ENABLED = False

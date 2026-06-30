# ACE-FP-EXPECT: clean
# CATEGORY: 34_reasoning_content_handling
# SOURCE: Mistral (Magistral) with reasoning_effort + mistralai python client
# WHY-CORRECT: when reasoning_effort is set, Mistral returns message.content as a LIST of chunks
#              ([ThinkChunk, TextChunk]), not a plain string; iterating the list by chunk.type and preserving the
#              ThinkChunk across turns is the documented Mistral reasoning idiom.
# EXPECTED-WRONG: engine flags iterating message.content (treating it as a list) or accessing chunk.type/.thinking
#                 as a type error because an OpenAI-centric schema assumes message.content is always a str.
# CORRECT-VERDICT: no findings
"""Iterate Mistral message.content as [ThinkChunk, TextChunk] under reasoning_effort."""

from mistralai import Mistral

client = Mistral(api_key="${MISTRAL_API_KEY}")


def solve(problem: str) -> tuple[str, str]:
    """Return (thinking, answer) by iterating the content chunk list.

    Args:
        problem: The problem to reason about.

    Returns:
        A tuple of (thinking_text, answer_text).
    """
    response = client.chat.complete(
        model="magistral-medium-latest",
        reasoning_effort="medium",
        messages=[{"role": "user", "content": problem}],
    )

    message = response.choices[0].message
    thinking_text = ""
    answer_text = ""

    # Under reasoning_effort, content is a list of typed chunks, not a string.
    for chunk in message.content:
        if chunk.type == "thinking":
            for piece in chunk.thinking:
                thinking_text += piece.text
        elif chunk.type == "text":
            answer_text += chunk.text

    return thinking_text, answer_text


if __name__ == "__main__":
    thoughts, answer = solve("A bat and ball cost $1.10; the bat costs $1 more. How much is the ball?")
    print(answer)

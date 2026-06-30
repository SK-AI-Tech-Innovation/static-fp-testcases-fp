# ACE-FP-EXPECT: clean
# CATEGORY: 31_prompt_caching_and_context
# SOURCE: OpenAI Python SDK (`openai`) conversation summary memory
# WHY-CORRECT: when the running transcript grows past a turn budget, the oldest turns are folded
#              into a single rolling summary and dropped, keeping recent turns verbatim. This is a
#              standard, correct memory-compression strategy to stay within budget.
# EXPECTED-WRONG: dated skill pack flags "you lose conversation history" or "no vector memory",
#                 missing that summarization is the intended, lossy-by-design budget strategy.
# CORRECT-VERDICT: no findings
"""Summarize old turns into a rolling memory to stay within a turn budget."""
from openai import OpenAI

client = OpenAI()
MAX_TURNS = 10


def summarize_old(turns: list[dict]) -> str:
    transcript = "\n".join(f"{t['role']}: {t['content']}" for t in turns)
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "Summarize this conversation in 3 sentences."},
            {"role": "user", "content": transcript},
        ],
    )
    return response.choices[0].message.content


def compact(history: list[dict]) -> list[dict]:
    if len(history) <= MAX_TURNS:
        return history
    keep = history[-MAX_TURNS:]
    summary = summarize_old(history[:-MAX_TURNS])
    return [{"role": "system", "content": f"Earlier conversation summary: {summary}"}] + keep


if __name__ == "__main__":
    hist = [{"role": "user", "content": f"msg {i}"} for i in range(25)]
    print(len(compact(hist)))

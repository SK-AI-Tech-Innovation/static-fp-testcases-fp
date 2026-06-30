# ACE-FP-EXPECT: clean
# CATEGORY: 28_streaming_async_variants
# SOURCE: httpx + raw Server-Sent Events (manual SSE parsing of an OpenAI-compatible /chat/completions endpoint)
# WHY-CORRECT: SSE frames arrive as `data: <json>` lines terminated by `data: [DONE]`; iterating response lines, stripping the "data:" prefix, breaking on [DONE], and json.loads on the rest is the correct raw-SSE contract.
# EXPECTED-WRONG: engine may claim the whole body should be response.json(), or flag "[DONE]" / empty lines as a parse bug.
# CORRECT-VERDICT: no findings
"""Parse raw SSE data lines from an OpenAI-compatible streaming endpoint."""

import json
import os

import httpx

API_URL = "https://api.openai.com/v1/chat/completions"


def stream_raw(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
    }

    out = []
    with httpx.stream("POST", API_URL, headers=headers, json=payload, timeout=60) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line or not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            event = json.loads(data)
            delta = event["choices"][0]["delta"]
            piece = delta.get("content")
            if piece:
                out.append(piece)

    return "".join(out)


if __name__ == "__main__":
    print(stream_raw("Count to five."))

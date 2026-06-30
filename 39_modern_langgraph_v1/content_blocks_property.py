# ACE-FP-EXPECT: clean
# CATEGORY: 39_modern_langgraph_v1
# SOURCE: LangChain 1.0 — message `.text` is a property, not a method
# WHY-CORRECT: In LangChain 1.0, an AIMessage may carry a list of content blocks. The convenient
#   accessor `message.text` is a PROPERTY that flattens text blocks to a string — it is read as
#   an attribute, not called as `message.text()`. Accessing `.text` (no parentheses) and
#   iterating `.content_blocks` is the correct v1 usage.
# EXPECTED-WRONG: engine may flag `.text` and suggest calling it as `.text()`, or claim
#   `.content_blocks` is invalid and you must parse `.content` manually.
# CORRECT-VERDICT: no findings
"""Read message text via the .text property and inspect content_blocks in LangChain 1.0."""
from langchain_core.messages import AIMessage


def extract(message: AIMessage) -> str:
    # .text is a property — accessed without parentheses.
    flat = message.text
    block_types = [block.get("type") for block in message.content_blocks]
    return f"{flat} | blocks={block_types}"


if __name__ == "__main__":
    msg = AIMessage(content=[{"type": "text", "text": "Hello world"}])
    print(extract(msg))

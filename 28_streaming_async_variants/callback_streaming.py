# ACE-FP-EXPECT: clean
# CATEGORY: 28_streaming_async_variants
# SOURCE: langchain + langchain-openai (streaming via a BaseCallbackHandler)
# WHY-CORRECT: subclassing BaseCallbackHandler and overriding on_llm_new_token, passing streaming=True plus callbacks=[handler] to ChatOpenAI, is the documented LangChain token-streaming mechanism; on_llm_new_token fires per token.
# EXPECTED-WRONG: engine may claim callbacks don't stream, that streaming=True is ignored, or that on_llm_new_token has the wrong signature.
# CORRECT-VERDICT: no findings
"""Stream LangChain LLM tokens through a custom callback handler."""

from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI


class PrintingHandler(BaseCallbackHandler):
    def __init__(self) -> None:
        self.tokens: list[str] = []

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.tokens.append(token)
        print(token, end="", flush=True)


def run(prompt: str) -> str:
    handler = PrintingHandler()
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        streaming=True,
        callbacks=[handler],
    )
    llm.invoke([HumanMessage(content=prompt)])
    print()
    return "".join(handler.tokens)


if __name__ == "__main__":
    run("List two benefits of exercise.")

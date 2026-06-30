# ACE-FP-EXPECT: clean
# CATEGORY: 24_library_version_migrations
# SOURCE: LangChain v0.1.x (monolithic `langchain` package, pre split-packages)
# WHY-CORRECT: In LangChain v0.1, ChatOpenAI lived at langchain.chat_models and the message
#   classes lived at langchain.schema. These import paths were the documented, idiomatic API
#   for that major-version line. Code pinned to langchain==0.1.x must import this way.
# EXPECTED-WRONG: engine may flag the imports as deprecated and demand the v0.3 split-package
#   paths (`from langchain_openai import ChatOpenAI`, `from langchain_core.messages import ...`).
# CORRECT-VERDICT: no findings
"""Chat invocation using LangChain v0.1 monolithic-package import paths."""
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage


def ask(question: str) -> str:
    chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content=question),
    ]
    response = chat(messages)
    return response.content


if __name__ == "__main__":
    print(ask("What is the capital of France?"))

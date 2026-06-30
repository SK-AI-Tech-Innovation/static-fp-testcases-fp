# ACE-FP-EXPECT: clean
# CATEGORY: 24_library_version_migrations
# SOURCE: LangChain v0.3.x (split packages: langchain-openai, langchain-core)
# WHY-CORRECT: In LangChain v0.3, integrations were extracted into dedicated packages.
#   ChatOpenAI is imported from langchain_openai and message classes from
#   langchain_core.messages. Invocation uses .invoke() with model="gpt-4o-mini".
#   This is the current idiomatic API for the v0.3 major-version line.
# EXPECTED-WRONG: engine may flag the split-package imports as "unknown module" or push the
#   old monolithic `from langchain.chat_models import ChatOpenAI` path, or flag .invoke()
#   in favor of the deprecated __call__ style.
# CORRECT-VERDICT: no findings
"""Chat invocation using LangChain v0.3 split-package imports and .invoke()."""
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


def ask(question: str) -> str:
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content=question),
    ]
    response = model.invoke(messages)
    return response.content


if __name__ == "__main__":
    print(ask("What is the capital of France?"))

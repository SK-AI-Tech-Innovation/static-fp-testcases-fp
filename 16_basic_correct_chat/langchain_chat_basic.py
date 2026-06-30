# ACE-FP-EXPECT: clean
# CATEGORY: 16_basic_correct_chat
# SOURCE: LangChain (`langchain-openai`) `ChatOpenAI.invoke`
# WHY-CORRECT: minimal idiomatic LangChain call — ChatOpenAI model, invoke with a list of
#              message objects, reply read from the returned AIMessage's .content. Complete.
# EXPECTED-WRONG: engine suggests "use a chain / output parser / prompt template" though a bare
#                 invoke is perfectly valid for a single message
# CORRECT-VERDICT: no findings
"""Ask a model a single question via LangChain's ChatOpenAI."""
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4.1-mini")


def ask(question: str) -> str:
    response = llm.invoke(
        [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content=question),
        ]
    )
    return response.content


if __name__ == "__main__":
    print(ask("What year did the first moon landing happen?"))

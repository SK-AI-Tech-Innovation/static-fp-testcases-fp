# ACE-FP-EXPECT: clean
# CATEGORY: 20_basic_correct_prompting
# SOURCE: LangChain `ChatPromptTemplate.from_messages` piped into a chat model
# WHY-CORRECT: system + human messages declared as a template with named input variables, composed
#              with the model via the LCEL pipe, and invoked with a variable dict. Textbook usage.
# EXPECTED-WRONG: engine suggests "use f-strings" or "add output parser" — neither is a defect for a
#                 simple templated chat chain.
# CORRECT-VERDICT: no findings
"""Build a templated chat chain with ChatPromptTemplate and a model."""
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that answers in {language}."),
        ("human", "{question}"),
    ]
)

model = ChatOpenAI(model="gpt-4.1-mini")
chain = prompt | model


def ask(question: str, language: str = "English") -> str:
    response = chain.invoke({"question": question, "language": language})
    return response.content


if __name__ == "__main__":
    print(ask("What is the speed of light?"))

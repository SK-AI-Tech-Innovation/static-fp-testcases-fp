# ACE-FP-EXPECT: clean
# CATEGORY: 16_basic_correct_chat
# SOURCE: OpenAI Python SDK (`openai`) `AzureOpenAI` client
# WHY-CORRECT: correct Azure usage — endpoint + api_version + deployment name passed as the
#              `model` argument (Azure convention), reply read from choices[0].message.content.
# EXPECTED-WRONG: engine flags "model looks like a deployment, not a model id" (correct for Azure)
# CORRECT-VERDICT: no findings
"""Ask an Azure OpenAI deployment a single question."""
import os

from openai import AzureOpenAI

client = AzureOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version="2024-10-21",
)


def ask(question: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-deployment",
        messages=[{"role": "user", "content": question}],
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    print(ask("Summarize photosynthesis in one sentence."))

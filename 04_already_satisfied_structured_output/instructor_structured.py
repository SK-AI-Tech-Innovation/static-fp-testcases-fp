# ACE-FP-EXPECT: clean
# CATEGORY: 04_already_satisfied_structured_output
# SOURCE: `instructor` patched OpenAI client with `response_model=Pydantic`
# WHY-CORRECT: instructor enforces the schema and re-validates/re-asks on failure; the
#              returned object is a validated Pydantic instance, no manual JSON parsing
# EXPECTED-WRONG: engine flags "free-text parsing / not using structured output" because the
#                 call goes through `client.chat.completions.create(...)` and there is no
#                 visible `response_format` / `.parse(...)` it recognizes
# CORRECT-VERDICT: no findings
"""Extract structured user details with an instructor-patched client (schema-enforced)."""
import instructor
from openai import OpenAI
from pydantic import BaseModel, Field

client = instructor.from_openai(OpenAI())


class UserDetail(BaseModel):
    name: str = Field(description="Full name of the person")
    age: int = Field(ge=0, le=130, description="Age in years")
    email: str | None = Field(default=None, description="Email if mentioned")


def extract_user(text: str) -> UserDetail:
    # response_model makes instructor validate the output and retry on schema errors;
    # the return value is already a UserDetail, never a raw string.
    return client.chat.completions.create(
        model="gpt-4.1-mini",
        response_model=UserDetail,
        max_retries=3,
        messages=[
            {"role": "system", "content": "Extract the user's details from the text."},
            {"role": "user", "content": text},
        ],
    )

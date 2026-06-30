# ACE-FP-EXPECT: clean
# CATEGORY: edge_cases/encoding
# LANGUAGE: python
# SOURCE: Correct, idiomatic LLM code that uses non-ASCII (Korean) identifiers and prompt text
# WHY-CORRECT: Uses schema-enforced structured output and a clean call; non-ASCII identifiers/strings are valid Python and must not confuse the engine
# EXPECTED-WRONG: The engine mis-handles the UTF-8 identifiers/strings, garbles current_code, or flags a non-issue
# CORRECT-VERDICT: no findings (the code is correct); no encoding-induced hallucination
"""한글 식별자와 프롬프트를 쓰는 정상적인 구조화 출력 코드 — 엔진이 인코딩에 흔들리면 안 됨."""

from openai import OpenAI
from pydantic import BaseModel

고객문의 = OpenAI()


class 분류결과(BaseModel):
    분류: str
    긴급도: int


def 문의_분류(문의내용: str) -> 분류결과:
    응답 = 고객문의.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[{"role": "user", "content": f"다음 문의를 분류하세요:\n{문의내용}"}],
        response_format=분류결과,
    )
    return 응답.choices[0].message.parsed

# ACE-FP-EXPECT: clean
# CATEGORY: 05_already_satisfied_retry_fallback
# SOURCE: litellm
# WHY-CORRECT: LiteLLM Router is configured with a model_list, num_retries, request timeout, and explicit fallbacks, so retries and provider failover are fully handled by the router.
# EXPECTED-WRONG: missing retry / missing fallback handling around the LLM call
# CORRECT-VERDICT: no findings
"""Route chat completions through a LiteLLM Router with retries and fallbacks."""

from litellm import Router

router = Router(
    model_list=[
        {
            "model_name": "primary-gpt",
            "litellm_params": {
                "model": "azure/gpt-4o",
                "api_base": "https://primary.openai.azure.com",
                "api_key": "os.environ/AZURE_API_KEY",
                "timeout": 30,
            },
        },
        {
            "model_name": "backup-gpt",
            "litellm_params": {
                "model": "openai/gpt-4o-mini",
                "api_key": "os.environ/OPENAI_API_KEY",
                "timeout": 30,
            },
        },
    ],
    num_retries=3,
    retry_after=2,
    timeout=30,
    fallbacks=[{"primary-gpt": ["backup-gpt"]}],
    allowed_fails=2,
    cooldown_time=60,
)


def summarize(text: str) -> str:
    """Summarize text; router retries and falls back to the backup model on failure."""
    response = router.completion(
        model="primary-gpt",
        messages=[
            {"role": "system", "content": "Summarize the user's text in one sentence."},
            {"role": "user", "content": text},
        ],
    )
    return response.choices[0].message.content

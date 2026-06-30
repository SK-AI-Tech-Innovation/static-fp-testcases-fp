# ACE-FP-EXPECT: clean
# CATEGORY: 27_provider_abstraction
# SOURCE: litellm.Router
# WHY-CORRECT: LiteLLM Router load-balances across a model_list of provider deployments and exposes a unified .completion call. The model_list schema (model_name alias + litellm_params) and routing_strategy are the documented Router API.
# EXPECTED-WRONG: engine may flag the heterogeneous provider model_list or the alias-based .completion call as an inconsistent/unknown client setup.
"""Multi-provider load balancing through a LiteLLM Router."""

import os

from litellm import Router


def build_router() -> Router:
    """Construct a Router that balances one logical model across providers."""
    model_list = [
        {
            "model_name": "assistant",
            "litellm_params": {
                "model": "anthropic/claude-sonnet-4-5",
                "api_key": os.environ["ANTHROPIC_API_KEY"],
            },
        },
        {
            "model_name": "assistant",
            "litellm_params": {
                "model": "openai/gpt-4o",
                "api_key": os.environ["OPENAI_API_KEY"],
            },
        },
    ]
    return Router(model_list=model_list, routing_strategy="simple-shuffle")


def ask(prompt: str) -> str:
    """Send a prompt to the 'assistant' alias and return the chosen reply."""
    router = build_router()
    response = router.completion(
        model="assistant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
    )
    return response["choices"][0]["message"]["content"]


if __name__ == "__main__":
    print(ask("What is a write-ahead log?"))

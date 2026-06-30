# ACE-FP-EXPECT: clean
# CATEGORY: 06_already_satisfied_observability
# SOURCE: OpenTelemetry GenAI semantic-convention spans around an LLM call
# WHY-CORRECT: the call is wrapped in a span using the `gen_ai.*` semantic conventions and
#              records request model, response model, and input/output token counts — this is
#              the canonical, vendor-neutral way to make an LLM call observable
# EXPECTED-WRONG: engine flags "missing observability/token tracking" because it looks for a
#                 logger or a Langfuse/W&B handler and does not recognize raw OTel span attrs
# CORRECT-VERDICT: no findings
"""Instrument a chat completion with OpenTelemetry GenAI semantic-convention spans."""
from openai import OpenAI
from opentelemetry import trace
from opentelemetry.semconv._incubating.attributes import gen_ai_attributes as ga

tracer = trace.get_tracer(__name__)
client = OpenAI()


def complete(prompt: str, model: str = "gpt-4.1-mini") -> str:
    with tracer.start_as_current_span("chat gpt-4.1-mini") as span:
        span.set_attribute(ga.GEN_AI_SYSTEM, "openai")
        span.set_attribute(ga.GEN_AI_OPERATION_NAME, "chat")
        span.set_attribute(ga.GEN_AI_REQUEST_MODEL, model)

        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )

        span.set_attribute(ga.GEN_AI_RESPONSE_MODEL, resp.model)
        span.set_attribute(ga.GEN_AI_USAGE_INPUT_TOKENS, resp.usage.prompt_tokens)
        span.set_attribute(ga.GEN_AI_USAGE_OUTPUT_TOKENS, resp.usage.completion_tokens)
        return resp.choices[0].message.content

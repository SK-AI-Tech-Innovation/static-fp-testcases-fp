# ACE-FP-EXPECT: clean
# CATEGORY: 07_non_ai_false_detection
# SOURCE: Jinja2 HTML email/page templating (Environment + render)
# WHY-CORRECT: This renders HTML templates with variable substitution. The words
#              "template"/"render"/"context" describe web/email templating, not LLM
#              prompt templates. There is no model, no chat, no PromptTemplate.
# EXPECTED-WRONG: "template", "render", "PromptRenderer"-like names + autoescape config ->
#                 false "prompt template / prompt injection" detection -> spurious findings.
# CORRECT-VERDICT: no findings
"""Render transactional HTML emails with Jinja2. Web templating, not LLM prompts."""
from __future__ import annotations

from dataclasses import dataclass

from jinja2 import Environment, StrictUndefined, select_autoescape

WELCOME_TEMPLATE = """\
<!doctype html>
<html>
  <body>
    <h1>Welcome, {{ user.name }}!</h1>
    <p>Your account on {{ site }} is ready.</p>
    <ul>
    {% for item in next_steps %}
      <li>{{ item }}</li>
    {% endfor %}
    </ul>
  </body>
</html>
"""


@dataclass
class User:
    name: str
    email: str


def build_environment() -> Environment:
    """Create a Jinja environment with autoescaping and strict undefineds."""
    return Environment(
        autoescape=select_autoescape(["html", "xml"]),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )


def render_welcome_email(user: User, site: str, next_steps: list[str]) -> str:
    """Render the welcome email HTML for a newly registered user."""
    env = build_environment()
    template = env.from_string(WELCOME_TEMPLATE)
    return template.render(user=user, site=site, next_steps=next_steps)

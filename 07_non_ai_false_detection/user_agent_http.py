# ACE-FP-EXPECT: clean
# CATEGORY: 07_non_ai_false_detection
# SOURCE: requests.Session with a custom User-Agent header (plain HTTP client)
# WHY-CORRECT: "agent" here is the HTTP User-Agent string identifying the client to a
#              web server. There is no LLM agent, planner, tool-calling loop, or model.
#              It is an ordinary HTTP crawler/scraper.
# EXPECTED-WRONG: keyword "agent" (user_agent, AGENT, build_agent) -> false "AI agent
#                 pattern" detection -> spurious findings about agent loops / tool use.
# CORRECT-VERDICT: no findings
"""Polite HTTP fetcher that sets a descriptive User-Agent. No LLM agent involved."""
from __future__ import annotations

import time
from dataclasses import dataclass

import requests

DEFAULT_AGENT = (
    "AcmeCrawler/2.3 (+https://acme.example/bot; contact=ops@acme.example) "
    "python-requests"
)


@dataclass
class AgentConfig:
    """Configuration for the User-Agent string and rate limiting."""

    user_agent: str = DEFAULT_AGENT
    delay_seconds: float = 1.0
    timeout: float = 10.0


def build_agent_session(config: AgentConfig) -> requests.Session:
    """Create a session whose User-Agent header announces this crawler."""
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": config.user_agent,
            "Accept": "text/html,application/xhtml+xml",
        }
    )
    return session


def crawl(urls: list[str], config: AgentConfig | None = None) -> dict[str, int]:
    """Fetch each URL once, respecting a polite delay between requests."""
    config = config or AgentConfig()
    session = build_agent_session(config)
    status_by_url: dict[str, int] = {}
    for url in urls:
        response = session.get(url, timeout=config.timeout)
        status_by_url[url] = response.status_code
        time.sleep(config.delay_seconds)
    return status_by_url


def is_known_bot_agent(user_agent: str) -> bool:
    """Heuristic check whether a User-Agent header looks like a crawler."""
    needle = user_agent.lower()
    return any(token in needle for token in ("bot", "crawler", "spider"))

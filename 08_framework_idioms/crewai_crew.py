# ACE-FP-EXPECT: clean
# CATEGORY: 08_framework_idioms
# SOURCE: CrewAI canonical Agent / Task / Crew setup with sequential process
# WHY-CORRECT: This is idiomatic CrewAI: each Agent carries role/goal/backstory, each Task
#              has a description + expected_output and is bound to an agent, and Crew ties
#              agents + tasks together with Process.sequential. The framework owns the LLM
#              calls and orchestration; there is no manual prompt string concatenation or
#              raw client call to inspect.
# EXPECTED-WRONG: prompt/best-practice checks may flag "no system prompt / no explicit model
#                 call" because the LLM invocation is implicit in crew.kickoff(); role/goal
#                 strings can look like unguarded prompt injection points.
# CORRECT-VERDICT: no findings
"""Idiomatic CrewAI: role-based Agents, bound Tasks, and a sequential Crew."""
from __future__ import annotations

from crewai import Agent, Crew, Process, Task

researcher = Agent(
    role="Senior Research Analyst",
    goal="Uncover the most relevant facts about {topic}",
    backstory=(
        "You are a meticulous analyst who values primary sources and "
        "distinguishes evidence from speculation."
    ),
    allow_delegation=False,
    verbose=True,
)

writer = Agent(
    role="Technical Writer",
    goal="Turn research notes into a clear, accurate briefing",
    backstory="You write tight, well-structured prose for busy engineers.",
    allow_delegation=False,
    verbose=True,
)

research_task = Task(
    description="Research {topic} and list the five most important findings.",
    expected_output="A bulleted list of five findings, each with a source.",
    agent=researcher,
)

writing_task = Task(
    description="Using the research findings, write a one-paragraph briefing.",
    expected_output="A single, self-contained paragraph.",
    agent=writer,
    context=[research_task],
)

crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writing_task],
    process=Process.sequential,
    verbose=True,
)


def run(topic: str) -> str:
    """Kick off the crew for a given topic and return the final briefing."""
    result = crew.kickoff(inputs={"topic": topic})
    return str(result)

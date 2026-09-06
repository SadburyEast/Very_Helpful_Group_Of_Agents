"""Planner: breaks the task into search queries, or revises them using critic feedback."""
from pydantic import BaseModel, Field

from config import settings
from llm import worker_llm
from state import ResearchState


class QueryList(BaseModel):
    queries: list[str] = Field(description="Search queries, most important first.")


_SYSTEM_PROMPT = """You turn a research question into a short list of focused \
web search queries. If critique feedback is provided, revise the queries to \
address the specific gaps it names -- don't just repeat the same queries. \
If a similar past session is provided, don't re-plan queries for things it \
already covers well -- focus queries on what it's missing or what may be \
stale (e.g. anything time-sensitive)."""


def planner_node(state: ResearchState) -> dict:
    planner = worker_llm.with_structured_output(QueryList)

    prompt = f"Research task: {state['task']}\n"

    recalled = state.get("recalled_context") or []
    if recalled:
        top = recalled[0]
        prompt += (
            f"\nSimilar past session found (similarity {top['similarity']}, "
            f"from {top['timestamp']}):\n{top['final_answer'][:600]}\n"
        )

    if state.get("critique"):
        prompt += f"\nPrevious critique to address: {state['critique']}\n"
    prompt += f"\nGenerate at most {settings.max_search_queries} queries."

    result: QueryList = planner.invoke(
        [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
    )

    return {
        "queries": result.queries[: settings.max_search_queries],
        "iteration": state.get("iteration", 0) + 1,
    }

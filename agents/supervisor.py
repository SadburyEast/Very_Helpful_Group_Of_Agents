"""
Supervisor: the routing brain.

This is what makes the graph a real multi-agent system rather than a
chain with agent-shaped names on the nodes -- control flow is decided
at runtime by a typed LLM decision, not by a fixed edge list.

Uses .with_structured_output() so the routing decision is a validated
Pydantic object, not a string we regex out of free text. That matters
more than usual here: with a single local model doing double duty as
router, a malformed/rambling routing response is the most likely
failure mode, and structured output constrains generation directly
rather than hoping the prompt is obeyed.
"""
from typing import Literal

from pydantic import BaseModel, Field

from llm import supervisor_llm
from state import ResearchState

Route = Literal["planner", "search", "critic", "writer", "finalize"]


class RouteDecision(BaseModel):
    next: Route = Field(description="Which agent should act next.")
    reasoning: str = Field(description="One short sentence explaining the choice.")


_SYSTEM_PROMPT = """You are the supervisor of a research agent team. Given the \
current state, decide which agent acts next.

Team:
- planner: generates or revises search queries. Use first, and again after \
a critic says the draft is insufficient.
- search: runs the current queries against the web. Use right after planner.
- writer: drafts a cited answer from accumulated search results. Use after \
search has produced results.
- critic: reviews the draft and judges if it's sufficient. Use right after writer.
- finalize: ends the loop and produces the final answer. Use when the critic \
says the draft is sufficient, OR when iteration >= max_iterations (don't loop \
forever even if the critic is picky).

Route to exactly one agent per turn based on what's missing in the state. \
Prefer looping back through planner (not directly writer<->critic) when the \
critic found the draft insufficient -- new queries are usually what's \
actually missing, not just a rewrite of the same draft."""


def supervisor_node(state: ResearchState) -> dict:
    router = supervisor_llm.with_structured_output(RouteDecision)

    total_hops = state.get("total_hops", 0) + 1
    max_total_hops = state.get("max_total_hops", 10)

    status = f"""
task: {state['task']}
iteration: {state.get('iteration', 0)} / max {state.get('max_iterations', 2)}
supervisor hop: {total_hops} / max {max_total_hops}
queries planned: {state.get('queries', [])}
search_results collected: {len(state.get('search_results', []))}
draft_answer present: {bool(state.get('draft_answer'))}
critique: {state.get('critique', '(none yet)')}
is_sufficient: {state.get('is_sufficient', None)}
"""

    decision: RouteDecision = router.invoke(
        [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": status},
        ]
    )

    next_agent = decision.next

    # Guardrail 1: the "normal" cap, tied to full planner-led loops.
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 2)
    if iteration >= max_iterations and next_agent != "finalize":
        next_agent = "finalize"

    # Guardrail 2: the ABSOLUTE cap. Trips regardless of path taken --
    # this is what actually bounds a supervisor that cycles between
    # agents (e.g. writer<->critic) without ever revisiting planner,
    # which would never trip guardrail 1. This one wins even over an
    # explicit "finalize" decision being one hop too late.
    if total_hops >= max_total_hops:
        next_agent = "finalize"

    return {
        "next_agent": next_agent,
        "total_hops": total_hops,
        "routing_trace": [
            {"iteration": iteration, "next": next_agent, "reasoning": decision.reasoning}
        ],
    }


def route_from_supervisor(state: ResearchState) -> str:
    """Conditional edge function -- just reads what supervisor_node decided."""
    return state["next_agent"]

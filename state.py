"""
Typed graph state.

Two reducer fields matter here:

  - search_results: Annotated[list, operator.add]
        Required for the Send-based parallel fan-out in agents/search.py --
        each parallel branch returns a partial list, and LangGraph merges
        them by concatenation instead of overwriting.

  - routing_trace: Annotated[list, operator.add]
        Not required for correctness, but makes the supervisor's decisions
        visible in the final state for the demo / debugging (each hop
        appends one entry rather than clobbering the last).
"""
from __future__ import annotations

from typing import Annotated, Literal, TypedDict
import operator


class SearchResult(TypedDict):
    query: str
    url: str
    title: str
    content: str


class RoutingDecision(TypedDict):
    iteration: int
    next: str
    reasoning: str


class RecalledSession(TypedDict):
    task: str
    final_answer: str
    timestamp: str
    similarity: float


class ResearchState(TypedDict):
    task: str

    # Written once by `recall` at graph start; read by planner and writer.
    # Empty list if nothing sufficiently similar was found in memory.
    recalled_context: list[RecalledSession]

    # Set by Planner, read by Search
    queries: list[str]

    # Written by parallel Search branches, merged via operator.add
    search_results: Annotated[list[SearchResult], operator.add]

    # Written by Writer, read by Critic and Finalize
    draft_answer: str

    # Written by Critic
    critique: str
    is_sufficient: bool

    iteration: int
    max_iterations: int

    # Absolute guardrail, independent of `iteration`: incremented on EVERY
    # supervisor visit, regardless of which agents were visited in between.
    # `iteration` only advances when planner runs, so a supervisor that
    # cycles writer<->critic (or any path that skips planner) would never
    # trip max_iterations -- total_hops always trips, no matter the path.
    total_hops: int
    max_total_hops: int

    # Written by Finalize -- the terminal output
    final_answer: str

    # Written by Supervisor each hop; drives the conditional edge
    next_agent: Literal["planner", "search", "critic", "writer", "finalize"]
    routing_trace: Annotated[list[RoutingDecision], operator.add]

"""
CLI entry point.

python main.py "your research question"

Uses app.stream() rather than app.invoke() so each agent's hop prints as
it happens -- useful for a live demo where "watch it route between
agents in real time" is the whole point, and also just makes the
30-90s sequential runtime (see config.py hardware note) feel active
rather than like a hang.
"""
import asyncio
import sys
import uuid

from langgraph.errors import GraphRecursionError

from config import settings, validate_settings
from graph import app
from tools import TavilyAuthError


async def run(question: str) -> str:
    validate_settings(settings)

    initial_state = {
        "task": question,
        "recalled_context": [],
        "queries": [],
        "search_results": [],
        "draft_answer": "",
        "critique": "",
        "is_sufficient": False,
        "iteration": 0,
        "max_iterations": settings.max_iterations,
        "total_hops": 0,
        "max_total_hops": settings.max_total_hops,
        "final_answer": "",
        "next_agent": "planner",
        "routing_trace": [],
    }

    # recursion_limit is guardrail #3: an independent backstop enforced
    # by LangGraph itself (counts every superstep, including parallel
    # search branches), in case a future edit breaks total_hops/iteration
    # logic in supervisor.py. Should never actually be the one that
    # trips in normal operation -- if it does, that's a bug upstream of it.
    config = {
        "configurable": {"thread_id": str(uuid.uuid4())},
        "recursion_limit": settings.recursion_limit,
    }

    final_answer = ""
    try:
        async for step in app.astream(initial_state, config=config):
            for node_name, node_output in step.items():
                print(f"\n--- {node_name} ---")
                if node_name == "recall":
                    recalled = node_output.get("recalled_context", [])
                    if recalled:
                        print(f"  found {len(recalled)} similar past session(s), top similarity {recalled[0]['similarity']}")
                    else:
                        print("  no sufficiently similar past session found")
                elif node_name == "persist_memory":
                    print("  session saved to long-term memory")
                elif node_name == "supervisor" and node_output.get("routing_trace"):
                    last = node_output["routing_trace"][-1]
                    hop = node_output.get("total_hops", "?")
                    print(f"  hop {hop}/{settings.max_total_hops} -> routing to: {last['next']}  ({last['reasoning']})")
                elif node_name == "planner":
                    print(f"  queries: {node_output.get('queries')}")
                elif node_name == "fetch_query":
                    n = len(node_output.get("search_results", []))
                    print(f"  fetched {n} result(s)")
                elif node_name == "writer":
                    print(f"  draft: {node_output.get('draft_answer', '')[:200]}...")
                elif node_name == "critic":
                    print(f"  sufficient: {node_output.get('is_sufficient')}  |  critique: {node_output.get('critique')}")
                elif node_name == "finalize":
                    final_answer = node_output.get("final_answer", "")
                    print(f"  final_answer length: {len(final_answer)} chars")
    except TavilyAuthError as exc:
        print(f"\n[FATAL] {exc}")
        sys.exit(1)
    except GraphRecursionError:
        print(
            f"\n[FATAL] Hit LangGraph's recursion_limit ({settings.recursion_limit}) "
            f"without finishing. This means the supervisor.py hop-count guardrail "
            f"(max_total_hops={settings.max_total_hops}) failed to trip before "
            f"this backstop did -- that's a bug worth investigating, not expected "
            f"behavior. Check routing_trace in the last state for what looped."
        )
        sys.exit(1)

    return final_answer


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python main.py "your research question"')
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    answer = asyncio.run(run(question))
    print("\n=== FINAL ANSWER ===\n")
    print(answer)

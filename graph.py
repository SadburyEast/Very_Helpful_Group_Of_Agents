"""
Graph wiring.

Shape:

              START
                |
                v
             recall              (memory lookup, no LLM, deterministic)
                |
                v
         +-------------+
    +--->|  supervisor |<---------------------------+
    |    +-------------+                             |
    |     next_agent = planner/search/critic/writer/finalize
    |           |
    |   +-------+-------+-------------+-------------+
    |   v               v             v             v
    | planner   search_dispatch    critic         writer
    |   |         (Send fanout)      |               |
    |   |               v            |               |
    |   |        fetch_query (xN,    |               |
    |   |         parallel)          |               |
    |   |               |            |               |
    +---+---------------+------------+---------------+
                                                       |
                                                    finalize
                                                       |
                                                       v
                                               persist_memory  (memory write, no LLM)
                                                       |
                                                      END

Every worker returns to `supervisor` -- that's what makes this a routed
multi-agent graph rather than a fixed pipeline with relabeled nodes.
`recall`/`persist_memory` are deterministic bookends, not something the
supervisor decides about -- there's nothing ambiguous about "check
memory first, save memory last," so they're plain fixed edges rather
than routed through the LLM.
"""
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from agents.critic import critic_node
from agents.finalize import finalize_node
from agents.persist_memory import persist_memory_node
from agents.planner import planner_node
from agents.recall import recall_node
from agents.search import fanout_to_queries, fetch_query_node, search_dispatch_node
from agents.supervisor import route_from_supervisor, supervisor_node
from agents.writer import writer_node
from state import ResearchState


def build_graph():
    graph = StateGraph(ResearchState)

    graph.add_node("recall", recall_node)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("planner", planner_node)
    graph.add_node("search_dispatch", search_dispatch_node)
    graph.add_node("fetch_query", fetch_query_node)
    graph.add_node("critic", critic_node)
    graph.add_node("writer", writer_node)
    graph.add_node("finalize", finalize_node)
    graph.add_node("persist_memory", persist_memory_node)

    graph.add_edge(START, "recall")
    graph.add_edge("recall", "supervisor")

    graph.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "planner": "planner",
            "search": "search_dispatch",
            "critic": "critic",
            "writer": "writer",
            "finalize": "finalize",
        },
    )

    # Send-based parallel fan-out: search_dispatch has no fixed edge,
    # instead fanout_to_queries dynamically returns one Send per query.
    graph.add_conditional_edges("search_dispatch", fanout_to_queries, ["fetch_query"])

    # Every worker reports back to the supervisor for the next routing
    # decision -- this is the loop that makes it agentic rather than linear.
    graph.add_edge("planner", "supervisor")
    graph.add_edge("fetch_query", "supervisor")
    graph.add_edge("critic", "supervisor")
    graph.add_edge("writer", "supervisor")

    graph.add_edge("finalize", "persist_memory")
    graph.add_edge("persist_memory", END)

    return graph.compile(checkpointer=MemorySaver())


app = build_graph()

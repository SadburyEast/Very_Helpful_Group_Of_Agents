"""
Search: the one part of the graph with genuine parallelism.

Deliberately has NO LLM call -- it's pure tool execution, so fanning it
out across queries costs nothing on the GPU. Uses LangGraph's Send API
to dispatch one `fetch_query` invocation per query; each runs the
(async) Tavily call concurrently, and results merge back into
`search_results` via the operator.add reducer declared in state.py.

`search_dispatch` itself is a no-op pass-through node -- it exists only
because add_conditional_edges needs a node to hang the Send-returning
function off of. All the real work happens in fetch_query.
"""
from langgraph.types import Send

from state import ResearchState
from tools import tavily_search


def search_dispatch_node(state: ResearchState) -> dict:
    return {}


def fanout_to_queries(state: ResearchState) -> list[Send]:
    queries = state.get("queries") or []
    if not queries:
        # Nothing planned -- shouldn't normally happen since planner runs
        # first, but fail safe rather than deadlock the graph.
        return [Send("fetch_query", {"query": state["task"]})]
    return [Send("fetch_query", {"query": q}) for q in queries]


async def fetch_query_node(state: dict) -> dict:
    """
    Runs as one parallel branch per query (via Send). `state` here is
    just {"query": ...} -- the partial payload Send dispatched, not the
    full graph state.
    """
    results = await tavily_search(state["query"])
    return {"search_results": results}

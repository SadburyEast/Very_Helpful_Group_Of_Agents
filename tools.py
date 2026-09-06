"""
Tool wrappers, isolated from agent/node logic.

tavily_search is async so the parallel Send-based fan-out in
agents/search.py gets real concurrency -- these are network calls,
not GPU-bound, so this is the one part of the system that genuinely
runs in parallel rather than queueing.
"""
from tavily import AsyncTavilyClient

from config import settings
from state import SearchResult

_client: AsyncTavilyClient | None = None

_AUTH_ERROR_MARKERS = ("401", "403", "unauthorized", "invalid api key", "invalid_api_key")


class TavilyAuthError(RuntimeError):
    """
    Raised (not swallowed) on what looks like a bad/invalid API key.

    Deliberately NOT caught into a placeholder [search error] result like
    other failures -- an invalid key means every single query this run
    will fail identically, so looping the graph through several more
    hops on worthless "results" just burns time before it eventually
    hits the hop-count guardrail. Failing fast here is faster AND
    clearer than waiting for that guardrail to trip.
    """


def _get_client() -> AsyncTavilyClient:
    global _client
    if _client is None:
        _client = AsyncTavilyClient(api_key=settings.tavily_api_key)
    return _client


async def tavily_search(query: str) -> list[SearchResult]:
    """Run a single query through Tavily and return normalized results."""
    client = _get_client()
    try:
        response = await client.search(
            query=query,
            max_results=settings.tavily_max_results_per_query,
        )
    except Exception as exc:  # noqa: BLE001
        message = str(exc).lower()
        if any(marker in message for marker in _AUTH_ERROR_MARKERS):
            raise TavilyAuthError(
                f"Tavily rejected the API key (looks like an auth error): {exc}. "
                f"Check TAVILY_API_KEY in your .env."
            ) from exc

        # Non-auth failures (rate limit, timeout, etc.) -- surface as a
        # result rather than crash the branch, since these are often
        # transient/per-query rather than session-wide.
        return [
            SearchResult(
                query=query,
                url="",
                title="[search error]",
                content=f"Search failed for query {query!r}: {exc}",
            )
        ]

    return [
        SearchResult(
            query=query,
            url=item.get("url", ""),
            title=item.get("title", ""),
            content=item.get("content", ""),
        )
        for item in response.get("results", [])
    ]

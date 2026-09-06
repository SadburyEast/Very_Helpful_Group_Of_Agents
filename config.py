"""
Settings for the research agent.

Hardware note: developed against a single local GPU (RTX 3070, 8GB VRAM)
running Ollama. This means LLM calls are effectively serialized at the
hardware level regardless of how the graph is structured -- only ONE
generation can run at a time. The architecture accounts for this:

  - Search fan-out parallelizes Tavily HTTP calls (cheap, I/O-bound),
    NOT LLM calls.
  - MAX_ITERATIONS defaults to 2 (not 3+) to bound total sequential
    LLM hops for a live demo.
  - SUPERVISOR_MODEL / WORKER_MODEL can be split so lightweight routing
    decisions use a smaller/faster model than the writer, if desired.
    Defaults to the same model for simplicity.
"""
from dataclasses import dataclass, field
import os
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    tavily_api_key: str = field(default_factory=lambda: os.getenv("TAVILY_API_KEY", ""))
    ollama_base_url: str = field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))

    # NOTE: qwen3.5-instruct was never a real Ollama library tag -- only
    # community namespaces (e.g. sorc/qwen3.5-instruct) offer a non-thinking
    # build, and their 9b size is an 11GB Q8_0 quant that won't fit an 8GB
    # card. Using the official qwen3.5:9b (thinking-capable, 6.6GB q4_K_M)
    # instead -- this brings the thinking-field-empty bug back into play,
    # which is exactly what llm.py's response_text() fallback is for.
    supervisor_model: str = field(default_factory=lambda: os.getenv("SUPERVISOR_MODEL", "qwen3.5:9b"))
    worker_model: str = field(default_factory=lambda: os.getenv("WORKER_MODEL", "qwen3.5:9b"))

    # Kept tight deliberately -- single-GPU sequential LLM calls, see above.
    num_ctx: int = int(os.getenv("OLLAMA_NUM_CTX", "8192"))
    num_predict: int = int(os.getenv("OLLAMA_NUM_PREDICT", "2048"))

    max_iterations: int = int(os.getenv("MAX_ITERATIONS", "2"))
    max_search_queries: int = int(os.getenv("MAX_SEARCH_QUERIES", "3"))
    tavily_max_results_per_query: int = int(os.getenv("TAVILY_MAX_RESULTS", "4"))

    # --- Absolute guardrails against runaway supervisor routing ---
    # `max_iterations` only advances when planner_node runs -- but the
    # supervisor can legally route critic<->writer (or any other cycle)
    # without ever touching planner, which would never trip that cap.
    # `max_total_hops` counts EVERY supervisor visit regardless of path
    # and is the real circuit breaker. `recursion_limit` is a second,
    # independent backstop enforced by LangGraph itself, in case a
    # future edit breaks the hop-counting logic above it.
    max_total_hops: int = int(os.getenv("MAX_TOTAL_HOPS", "10"))
    recursion_limit: int = int(os.getenv("RECURSION_LIMIT", "30"))

    # Cross-session memory -- CPU embeddings, deliberately not through Ollama.
    # See memory.py module docstring for why.
    embedding_model: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))
    memory_dir: str = field(default_factory=lambda: os.getenv("MEMORY_DIR", "./.memory"))
    recall_top_k: int = int(os.getenv("RECALL_TOP_K", "3"))
    recall_similarity_threshold: float = float(os.getenv("RECALL_SIMILARITY_THRESHOLD", "0.6"))


# Ships in .env.example so `python main.py` fails clearly instead of a
# generic 401 from Tavily -- this exact placeholder is what gets left in
# by accident if someone copies .env.example to .env and forgets the key.
_PLACEHOLDER_TAVILY_KEY = "tvly-xxxxxxxxxxxxxxxxxxxx"


def validate_settings(settings: Settings) -> None:
    missing = []
    if not settings.tavily_api_key:
        missing.append("TAVILY_API_KEY")
    if missing:
        raise ValueError(
            f"Missing required settings: {', '.join(missing)}. "
            f"Copy .env.example to .env and fill it in."
        )

    if settings.tavily_api_key == _PLACEHOLDER_TAVILY_KEY:
        raise ValueError(
            "TAVILY_API_KEY is still set to the placeholder value from "
            ".env.example. Get a real key from https://tavily.com and put "
            "it in your .env file."
        )

    if not settings.tavily_api_key.startswith("tvly-"):
        raise ValueError(
            f"TAVILY_API_KEY doesn't look like a real Tavily key (expected "
            f"it to start with 'tvly-'). Double check your .env file."
        )


settings = Settings()

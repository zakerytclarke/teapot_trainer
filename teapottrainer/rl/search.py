"""Configurable web-search tools for RL.

Default backend is **llmsearchindex** (local FineWeb+Wikipedia index) via
``pip install llmsearchindex``. Brave / Exa / OpenAI-compatible ``llm_search``
remain available but are opt-in.

Each factory returns a plain Python function whose name/docstring/type hints
become the Hugging Face tool schema. Knobs come from ``SearchSettings``.

Environment variables (optional backends):

* ``BRAVE_API_KEY`` / ``BRAVE_SEARCH_API_KEY``
* ``EXA_API_KEY``
* ``LLM_SEARCH_API_KEY`` / ``OPENAI_API_KEY``
* ``LLM_SEARCH_BASE_URL`` (OpenAI-compatible chat completions root)
* ``LLM_SEARCH_MODEL``
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable, Optional

from pydantic import BaseModel, ConfigDict

ToolFn = Callable[..., Any]

# Shared LLMIndex instance — download/load is expensive.
_LLMSEARCH_INDEX = None
_LLMSEARCH_IMPORT_ERROR: Exception | None = None


def _env(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


class SearchSettings(BaseModel):
    """Shared knobs for search tool backends."""

    # Default / primary: local llmsearchindex (https://pypi.org/project/llmsearchindex/)
    llmsearchindex_top_k: int = 5
    llmsearchindex_rerank: bool = False

    brave_api_key: Optional[str] = None
    brave_count: int = 5
    brave_endpoint: str = "https://api.search.brave.com/res/v1/web/search"

    exa_api_key: Optional[str] = None
    exa_num_results: int = 5
    exa_endpoint: str = "https://api.exa.ai/search"

    llm_search_api_key: Optional[str] = None
    llm_search_base_url: Optional[str] = None
    llm_search_model: str = "gpt-4o-mini"
    llm_search_system: str = (
        "You are a web research assistant. Given a search query, return concise "
        "factual findings with source URLs when possible. Prefer short bullet points."
    )

    timeout_s: float = 20.0
    # When True, tools never load the index / hit APIs; they return stubs.
    mock: bool = False
    # Default provider list when ``build_search_tools`` is called without providers.
    # ``llmsearchindex`` only for now; pass brave/exa/llm_search explicitly later.
    default_providers: list[str] = ["llmsearchindex"]

    model_config = ConfigDict(extra="ignore")

    @classmethod
    def from_env(cls, **overrides: Any) -> "SearchSettings":
        """Build settings from process env, then apply keyword overrides."""
        data = {
            "brave_api_key": _env("BRAVE_API_KEY", "BRAVE_SEARCH_API_KEY"),
            "exa_api_key": _env("EXA_API_KEY"),
            "llm_search_api_key": _env("LLM_SEARCH_API_KEY", "OPENAI_API_KEY"),
            "llm_search_base_url": _env("LLM_SEARCH_BASE_URL", "OPENAI_BASE_URL"),
            "llm_search_model": _env("LLM_SEARCH_MODEL") or "gpt-4o-mini",
        }
        data.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**data)


def _get_llmsearch_index():
    """Lazy-load the shared ``LLMIndex`` (downloads ~GB on first use)."""
    global _LLMSEARCH_INDEX, _LLMSEARCH_IMPORT_ERROR
    if _LLMSEARCH_INDEX is not None:
        return _LLMSEARCH_INDEX
    if _LLMSEARCH_IMPORT_ERROR is not None:
        raise _LLMSEARCH_IMPORT_ERROR
    try:
        from llmsearchindex import LLMIndex  # type: ignore
    except Exception as exc:  # noqa: BLE001
        _LLMSEARCH_IMPORT_ERROR = RuntimeError(
            "llmsearchindex is not installed. Run: pip install llmsearchindex"
        )
        raise _LLMSEARCH_IMPORT_ERROR from exc
    _LLMSEARCH_INDEX = LLMIndex()
    return _LLMSEARCH_INDEX


def _http_json(
    method: str,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    body: dict | None = None,
    timeout_s: float = 20.0,
) -> Any:
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method.upper())
    req.add_header("Accept", "application/json")
    if body is not None:
        req.add_header("Content-Type", "application/json")
    for key, value in (headers or {}).items():
        req.add_header(key, value)
    try:
        with urllib.request.urlopen(req, timeout=float(timeout_s)) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace") if exc.fp else str(exc)
        raise RuntimeError(f"HTTP {exc.code} for {url}: {detail[:500]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"request failed for {url}: {exc}") from exc
    if not raw.strip():
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"raw": raw}


def _format_hits(hits: list[dict[str, str]], *, limit: int) -> str:
    if not hits:
        return "No results."
    lines = []
    for index, hit in enumerate(hits[: max(1, int(limit))], start=1):
        title = (hit.get("title") or "").strip() or "(untitled)"
        url = (hit.get("url") or "").strip()
        snippet = (hit.get("snippet") or "").strip()
        block = f"{index}. {title}"
        if url:
            block += f"\n   URL: {url}"
        if snippet:
            block += f"\n   {snippet}"
        lines.append(block)
    return "\n".join(lines)


def make_llmsearchindex(settings: SearchSettings | None = None, **overrides: Any) -> ToolFn:
    """Local web search via ``llmsearchindex`` (FineWeb + Wikipedia index)."""
    cfg = (settings or SearchSettings.from_env()).model_copy(update=overrides)

    def web_search(query: str, top_k: int | None = None) -> str:
        """Search a local web index (Wikipedia + FineWeb) for passages.

        Args:
            query: Search query string.
            top_k: Number of hits to return (defaults to configured top_k).
        """
        q = str(query or "").strip()
        if not q:
            return "ERROR: empty query"
        k = int(top_k) if top_k is not None else int(cfg.llmsearchindex_top_k)
        k = max(1, min(k, 20))
        if cfg.mock:
            return _format_hits(
                [
                    {
                        "title": f"Mock llmsearchindex hit for {q}",
                        "url": "https://example.com/llmsearchindex",
                        "snippet": f"Stub indexed passage about {q}.",
                    }
                ],
                limit=k,
            )
        try:
            index = _get_llmsearch_index()
            raw = index.search(q, top_k=k, rerank=bool(cfg.llmsearchindex_rerank))
        except Exception as exc:  # noqa: BLE001
            return f"ERROR: llmsearchindex failed: {exc}"
        hits = []
        for row in raw or []:
            if not isinstance(row, dict):
                continue
            snippet = str(row.get("snippet") or row.get("text") or "")[:500]
            hits.append(
                {
                    "title": str(row.get("title") or row.get("url") or "(untitled)"),
                    "url": str(row.get("url") or ""),
                    "snippet": snippet,
                }
            )
        return _format_hits(hits, limit=k)

    return web_search


def make_brave_search(settings: SearchSettings | None = None, **overrides: Any) -> ToolFn:
    """Brave Web Search API tool (opt-in; not used by default)."""
    cfg = (settings or SearchSettings.from_env()).model_copy(update=overrides)

    def brave_search(query: str, count: int | None = None) -> str:
        """Search the public web with the Brave Search API.

        Args:
            query: Search query string.
            count: Max results to return (defaults to the configured Brave count).
        """
        q = str(query or "").strip()
        if not q:
            return "ERROR: empty query"
        n = int(count) if count is not None else int(cfg.brave_count)
        n = max(1, min(n, 20))
        if cfg.mock:
            return _format_hits(
                [
                    {
                        "title": f"Mock Brave result for {q}",
                        "url": "https://example.com/brave",
                        "snippet": f"Stub snippet about {q}.",
                    }
                ],
                limit=n,
            )
        key = cfg.brave_api_key or _env("BRAVE_API_KEY", "BRAVE_SEARCH_API_KEY")
        if not key:
            return "ERROR: Brave API key missing (set brave_api_key or BRAVE_API_KEY)"
        params = urllib.parse.urlencode({"q": q, "count": str(n)})
        url = f"{cfg.brave_endpoint}?{params}"
        payload = _http_json(
            "GET",
            url,
            headers={"X-Subscription-Token": key},
            timeout_s=cfg.timeout_s,
        )
        hits = []
        for row in (payload.get("web") or {}).get("results") or []:
            if not isinstance(row, dict):
                continue
            hits.append(
                {
                    "title": str(row.get("title") or ""),
                    "url": str(row.get("url") or ""),
                    "snippet": str(row.get("description") or row.get("snippet") or ""),
                }
            )
        return _format_hits(hits, limit=n)

    return brave_search


def make_exa_search(settings: SearchSettings | None = None, **overrides: Any) -> ToolFn:
    """Exa neural search API tool."""
    cfg = (settings or SearchSettings.from_env()).model_copy(update=overrides)

    def exa_search(query: str, num_results: int | None = None) -> str:
        """Search the web with the Exa API.

        Args:
            query: Search query string.
            num_results: Max results to return (defaults to the configured Exa count).
        """
        q = str(query or "").strip()
        if not q:
            return "ERROR: empty query"
        n = int(num_results) if num_results is not None else int(cfg.exa_num_results)
        n = max(1, min(n, 20))
        if cfg.mock:
            return _format_hits(
                [
                    {
                        "title": f"Mock Exa result for {q}",
                        "url": "https://example.com/exa",
                        "snippet": f"Stub Exa passage about {q}.",
                    }
                ],
                limit=n,
            )
        key = cfg.exa_api_key or _env("EXA_API_KEY")
        if not key:
            return "ERROR: Exa API key missing (set exa_api_key or EXA_API_KEY)"
        payload = _http_json(
            "POST",
            cfg.exa_endpoint,
            headers={"x-api-key": key},
            body={
                "query": q,
                "numResults": n,
                "contents": {"text": {"maxCharacters": 400}},
            },
            timeout_s=cfg.timeout_s,
        )
        hits = []
        for row in payload.get("results") or []:
            if not isinstance(row, dict):
                continue
            text = row.get("text") or row.get("summary") or ""
            hits.append(
                {
                    "title": str(row.get("title") or ""),
                    "url": str(row.get("url") or ""),
                    "snippet": str(text)[:400],
                }
            )
        return _format_hits(hits, limit=n)

    return exa_search


def make_llm_search(settings: SearchSettings | None = None, **overrides: Any) -> ToolFn:
    """OpenAI-compatible chat-completions 'search' tool (Perplexity, proxies, etc.)."""
    cfg = (settings or SearchSettings.from_env()).model_copy(update=overrides)

    def llm_search(query: str) -> str:
        """Ask an LLM search endpoint for findings on a query.

        Args:
            query: Search / research query string.
        """
        q = str(query or "").strip()
        if not q:
            return "ERROR: empty query"
        if cfg.mock:
            return f"Mock LLM search findings for {q}. Source: https://example.com/llm"
        key = cfg.llm_search_api_key or _env("LLM_SEARCH_API_KEY", "OPENAI_API_KEY")
        base = (cfg.llm_search_base_url or _env("LLM_SEARCH_BASE_URL", "OPENAI_BASE_URL") or "").rstrip("/")
        if not key:
            return "ERROR: LLM search API key missing (set llm_search_api_key or LLM_SEARCH_API_KEY)"
        if not base:
            return "ERROR: LLM search base URL missing (set llm_search_base_url or LLM_SEARCH_BASE_URL)"
        endpoint = base if base.endswith("/chat/completions") else f"{base}/chat/completions"
        payload = _http_json(
            "POST",
            endpoint,
            headers={"Authorization": f"Bearer {key}"},
            body={
                "model": cfg.llm_search_model,
                "temperature": 0.2,
                "messages": [
                    {"role": "system", "content": cfg.llm_search_system},
                    {"role": "user", "content": q},
                ],
            },
            timeout_s=cfg.timeout_s,
        )
        choices = payload.get("choices") or []
        if not choices:
            return f"ERROR: empty LLM search response: {json.dumps(payload)[:400]}"
        message = (choices[0] or {}).get("message") or {}
        content = message.get("content")
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text") or ""))
                elif isinstance(item, str):
                    parts.append(item)
            content = "\n".join(p for p in parts if p)
        text = str(content or "").strip()
        return text or "ERROR: blank LLM search content"

    return llm_search


def build_search_tools(
    settings: SearchSettings | None = None,
    *,
    providers: list[str] | None = None,
    **overrides: Any,
) -> list[ToolFn]:
    """Return configured search tools.

    Default providers: ``[\"llmsearchindex\"]`` only. Pass
    ``providers=[\"brave\", \"exa\", ...]`` to opt into API backends.
    Aliases: ``llmsearchindex``, ``index``, ``web_search``.
    """
    cfg = (settings or SearchSettings.from_env()).model_copy(update=overrides)
    raw = providers if providers is not None else list(cfg.default_providers or ["llmsearchindex"])
    wanted = []
    for item in raw:
        key = str(item).strip().lower()
        if key in {"index", "web_search", "llmsearch", "llm_search_index"}:
            key = "llmsearchindex"
        wanted.append(key)
    tools: list[ToolFn] = []

    if "llmsearchindex" in wanted:
        tools.append(make_llmsearchindex(cfg))
    if "brave" in wanted:
        tools.append(make_brave_search(cfg))
    if "exa" in wanted:
        tools.append(make_exa_search(cfg))
    if "llm_search" in wanted:
        tools.append(make_llm_search(cfg))
    return tools

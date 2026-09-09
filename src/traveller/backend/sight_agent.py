"""
SightAgent
----------
A small LangGraph pipeline that, given an OpenTripMap "xid" (returned by
TravelAgent.get_top_sights), pulls structured details, searches the web
for extra context, and asks a local LLM (via Ollama) to write a short
curated travel description. Triggered when the user clicks a sight in
the UI, not when the sight list first loads - this keeps things fast and
only spends model/search calls on sights people actually want to read
about.

Requires:
- Ollama running locally with the model pulled, e.g.:
      ollama pull gemma3n:e4b
  (change the tag via OLLAMA_MODEL in .env if you're using a different one)
- OTM_API_KEY already set in .env (reused from the top-sights feature)

This is stateless per request - nothing is cached. If you start hitting
the same popular sights a lot, add a simple cache (e.g. a dict keyed by
xid, or a small SQLite table) in front of get_curated_sight().
"""

from __future__ import annotations

import os
from typing import List, Optional, TypedDict

import requests
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph

from agent import OTM_API_KEY

OTM_XID_URL = "https://api.opentripmap.com/0.1/en/places/xid/{xid}"

OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma3n:e4b")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

print(f"[sight_agent] Using Ollama model '{OLLAMA_MODEL}' at {OLLAMA_BASE_URL}")


class SightState(TypedDict, total=False):
    xid: str
    name: str
    lat: Optional[float]
    lon: Optional[float]
    image_url: Optional[str]
    wiki_extract: Optional[str]
    search_snippets: List[str]
    sources: List[str]
    description: str
    error: Optional[str]


def _fetch_details(state: SightState) -> SightState:
    """Node 1: pull structured facts (image, wiki extract, links) from OpenTripMap."""
    if not OTM_API_KEY:
        state["error"] = (
            "No OTM_API_KEY set. Get a free key at https://opentripmap.io/product "
            "and set it in your .env file."
        )
        return state

    resp = requests.get(
        OTM_XID_URL.format(xid=state["xid"]),
        params={"apikey": OTM_API_KEY},
        timeout=15,
    )
    if resp.status_code != 200:
        state["error"] = f"Couldn't load sight details ({resp.status_code})."
        return state

    data = resp.json()
    state["name"] = data.get("name") or state.get("name") or "Unknown sight"

    point = data.get("point") or {}
    state["lat"] = point.get("lat")
    state["lon"] = point.get("lon")

    preview = data.get("preview") or {}
    state["image_url"] = preview.get("source") or data.get("image")

    extracts = data.get("wikipedia_extracts") or {}
    state["wiki_extract"] = extracts.get("text")

    sources = []
    if data.get("wikipedia"):
        sources.append(data["wikipedia"])
    if data.get("url"):
        sources.append(data["url"])
    state["sources"] = sources

    return state


def _web_search(state: SightState) -> SightState:
    """Node 2: gather a few extra web snippets to give the model more to work with."""
    if state.get("error"):
        return state
    try:
        search = DuckDuckGoSearchResults(output_format="list", num_results=4)
        query = f"{state['name']} Bulgaria travel guide history"
        results = search.invoke(query)

        snippets = []
        sources = list(state.get("sources", []))
        for r in results:
            snippet = r.get("snippet") or r.get("body") or ""
            link = r.get("link") or r.get("href")
            if snippet:
                snippets.append(snippet)
            if link:
                sources.append(link)

        state["search_snippets"] = snippets
        state["sources"] = sources
    except Exception:
        # A flaky search shouldn't block curation - just proceed with what we have
        # (the Wikipedia extract from OpenTripMap, if any).
        state["search_snippets"] = []
    return state


def _curate(state: SightState) -> SightState:
    """Node 3: ask the local model to synthesize everything into a short description."""
    if state.get("error"):
        return state

    context_parts = []
    if state.get("wiki_extract"):
        context_parts.append(f"Wikipedia: {state['wiki_extract']}")
    for i, snippet in enumerate(state.get("search_snippets", []), start=1):
        context_parts.append(f"Web result {i}: {snippet}")

    context = "\n\n".join(context_parts) if context_parts else "No extra information found."

    prompt = (
        "You are a concise, trustworthy travel guide writer. Using ONLY the "
        "information given below, write a short curated description (3-4 "
        "sentences) of the sight for a traveler deciding whether to visit. "
        "Do not invent facts that aren't supported by the context. If the "
        "context is thin, keep the description general and say so briefly.\n\n"
        f"Sight name: {state['name']}\n\n"
        f"Context:\n{context}"
    )

    try:
        llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.4)
        response = llm.invoke(prompt)
        state["description"] = response.content.strip()
    except Exception as e:
        state["error"] = (
            f"Couldn't reach the local model '{OLLAMA_MODEL}' at {OLLAMA_BASE_URL}. "
            f"Is Ollama running and is the model pulled? ({e})"
        )

    return state


_graph = StateGraph(SightState)
_graph.add_node("fetch_details", _fetch_details)
_graph.add_node("web_search", _web_search)
_graph.add_node("curate", _curate)
_graph.set_entry_point("fetch_details")
_graph.add_edge("fetch_details", "web_search")
_graph.add_edge("web_search", "curate")
_graph.add_edge("curate", END)

sight_pipeline = _graph.compile()


def get_curated_sight(xid: str, fallback_name: str = "") -> dict:
    """Run the full pipeline for a single sight and return a clean, frontend-ready dict."""
    result = sight_pipeline.invoke({"xid": xid, "name": fallback_name})

    if result.get("error"):
        raise RuntimeError(result["error"])

    return {
        "name": result.get("name"),
        "image_url": result.get("image_url"),
        "description": result.get("description", ""),
        "sources": result.get("sources", [])[:3],
    }

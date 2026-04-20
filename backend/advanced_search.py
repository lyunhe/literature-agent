"""
Advanced multi-round search strategies with LLM planning, relevance scoring,
and iterative query refinement.
"""
import json
import re
from typing import Any

from .llm_client import llm_request
from .search import arxiv_search, ieee_search


POWER_SYSTEM_GLOSSARY = {
    "SCUC": ["Security Constrained Unit Commitment", "security-constrained unit commitment", "SCUC"],
    "UC": ["Unit Commitment", "unit commitment", "UC"],
    "并行计算": ["parallel computing", "GPU acceleration", "distributed computing", "high-performance computing", "HPC"],
    "Benders分解": ["Benders decomposition", "Benders' decomposition", "L-shaped method"],
    "电力系统": ["power system", "electrical grid", "power grid", "electricity system"],
    "优化": ["optimization", "mathematical programming", "mixed-integer programming", "MILP"],
}


def _safe_json_loads(text: str) -> Any:
    text = (text or "").strip()
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1]
            if text.startswith("json"):
                text = text[4:]
    text = text.strip()
    return json.loads(text)


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _paper_key(paper: dict) -> str:
    if paper.get("doi"):
        return f"doi:{paper['doi'].lower()}"
    if paper.get("arxiv_id"):
        return f"arxiv:{paper['arxiv_id']}"
    if paper.get("ieee_id"):
        return f"ieee:{paper['ieee_id']}"
    return f"title:{_normalize_text(paper.get('title', ''))}"


def _parse_number(text: str, default: float = 5.0) -> float:
    matches = re.findall(r"\d+(?:\.\d+)?", text or "")
    if not matches:
        return default
    try:
        return float(matches[0])
    except ValueError:
        return default


def _truncate(text: str, limit: int = 500) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[:limit] + "..."


def _build_glossary_text() -> str:
    return "\n".join(f"- {k}: {', '.join(v)}" for k, v in POWER_SYSTEM_GLOSSARY.items())


def generate_query_variations(user_query: str, num_variations: int = 3) -> list[str]:
    """
    Backward-compatible helper used by older flows and as a fallback.
    """
    glossary_text = _build_glossary_text()
    prompt = f"""Given this research query: "{user_query}"

Domain terminology reference:
{glossary_text}

Generate {num_variations} different search query variations in English that:
1. Use different combinations of technical terms from the glossary
2. Include specific methods or technical sub-topics when relevant
3. Focus on the core problem and adjacent research terminology
4. Are suitable for academic paper search

Return ONLY a JSON array of strings, no explanation."""

    try:
        resp = llm_request([{"role": "user", "content": prompt}], max_tokens=400, temperature=0.7)
        queries = _safe_json_loads(resp.choices[0].message.content)
        if isinstance(queries, list):
            clean_queries = [str(q).strip() for q in queries if str(q).strip()]
            if clean_queries:
                return clean_queries[:num_variations]
    except Exception as e:
        print(f"[Warning] Query generation failed: {e}, using fallback queries")

    return [
        user_query,
        f"{user_query} optimization methods",
        f"{user_query} survey review",
    ][:num_variations]


def plan_search_strategy(
    user_query: str,
    max_queries: int = 6,
    preferred_sources: list[str] | None = None,
) -> dict:
    """
    Ask the LLM to expand the topic into a structured search plan.
    """
    preferred_sources = preferred_sources or ["arxiv", "ieee"]
    glossary_text = _build_glossary_text()
    prompt = f"""You are planning a literature search workflow.

Research topic:
{user_query}

Available sources:
- arxiv: broad preprints, strong for methods and recent ML/CS work
- ieee: engineering venue search, useful for journal/conference papers if the API key is available

Domain terminology reference:
{glossary_text}

Return ONLY a JSON object with this schema:
{{
  "topic_summary": "1-3 sentence description of the topic and research intent",
  "expanded_topics": ["subtopic 1", "subtopic 2"],
  "core_keywords": ["keyword"],
  "method_keywords": ["method keyword"],
  "application_keywords": ["application keyword"],
  "target_venues": ["journal or conference name"],
  "target_authors": ["author name"],
  "queries": [
    {{
      "query": "search query text",
      "source": "arxiv or ieee",
      "rationale": "why this query is useful"
    }}
  ]
}}

Requirements:
1. Expand the user's topic before searching.
2. Identify worthwhile journals or conferences when possible.
3. Identify likely influential authors when possible.
4. Include a mix of broad and focused queries.
5. Use only these sources: {preferred_sources}.
6. Provide at most {max_queries} queries."""

    try:
        resp = llm_request([{"role": "user", "content": prompt}], max_tokens=1200, temperature=0.5)
        plan = _safe_json_loads(resp.choices[0].message.content)
        if isinstance(plan, dict) and isinstance(plan.get("queries"), list) and plan["queries"]:
            plan["queries"] = [
                {
                    "query": str(item.get("query", "")).strip(),
                    "source": str(item.get("source", "arxiv")).strip().lower(),
                    "rationale": str(item.get("rationale", "")).strip(),
                }
                for item in plan["queries"]
                if str(item.get("query", "")).strip()
            ][:max_queries]
            if plan["queries"]:
                return plan
    except Exception as e:
        print(f"[Warning] Search planning failed: {e}, using fallback plan")

    fallback_queries = generate_query_variations(user_query, num_variations=min(3, max_queries))
    return {
        "topic_summary": user_query,
        "expanded_topics": [],
        "core_keywords": [user_query],
        "method_keywords": [],
        "application_keywords": [],
        "target_venues": [],
        "target_authors": [],
        "queries": [
            {"query": q, "source": "arxiv", "rationale": "Fallback query variation"}
            for q in fallback_queries
        ],
    }


def _search_one_source(source: str, query: str, max_results: int) -> list[dict]:
    if source == "ieee":
        return ieee_search.search(query, max_results=max_results)
    return arxiv_search.search(query, max_results=max_results)


def _collect_papers_from_plan(plan: dict, per_query_limit: int = 8) -> tuple[list[dict], list[dict]]:
    all_papers: list[dict] = []
    diagnostics: list[dict] = []
    seen: set[str] = set()

    for item in plan.get("queries", []):
        query = item.get("query", "")
        source = item.get("source", "arxiv")
        if not query:
            continue

        print(f"[Search] {source}: {query[:100]}")
        results = _search_one_source(source, query, max_results=per_query_limit)
        diagnostics.append({
            "query": query,
            "source": source,
            "rationale": item.get("rationale", ""),
            "result_count": len(results),
        })

        for paper in results:
            if "error" in paper:
                diagnostics[-1]["error"] = paper["error"]
                continue
            enriched = dict(paper)
            enriched["search_query"] = query
            enriched["search_source"] = source
            key = _paper_key(enriched)
            if key in seen:
                continue
            seen.add(key)
            all_papers.append(enriched)

    return all_papers, diagnostics


def score_relevance(user_query: str, paper: dict, topic_summary: str = "") -> float:
    """
    Score a single paper 0-10. Kept for backward compatibility and fallback.
    """
    prompt = f"""Rate the relevance of this paper to the research topic on a scale of 0-10.

Research Topic: "{user_query}"
Topic Summary: "{topic_summary}"

Paper Title: {paper.get('title', 'N/A')}
Paper Abstract: {_truncate(paper.get('abstract', 'N/A'), 700)}

Return ONLY a single number between 0 and 10."""
    try:
        resp = llm_request([{"role": "user", "content": prompt}], max_tokens=20, temperature=0.2)
        score = _parse_number(resp.choices[0].message.content, default=5.0)
        return min(max(score, 0.0), 10.0)
    except Exception as e:
        print(f"[Warning] Scoring failed for '{paper.get('title', 'unknown')}': {e}")
        return 5.0


def batch_score_papers(user_query: str, plan: dict, papers: list[dict], top_k: int = 12) -> list[dict]:
    """
    Ask the LLM to score a batch of candidate papers and explain the fit briefly.
    """
    if not papers:
        return []

    candidates = papers[:top_k]
    payload = []
    for idx, paper in enumerate(candidates, start=1):
        payload.append({
            "id": idx,
            "title": paper.get("title", ""),
            "authors": paper.get("authors", ""),
            "year": paper.get("year"),
            "source": paper.get("source", ""),
            "query": paper.get("search_query", ""),
            "abstract": _truncate(paper.get("abstract", ""), 700),
        })

    prompt = f"""You are evaluating search results for a literature review.

Research Topic:
{user_query}

Topic Summary:
{plan.get("topic_summary", "")}

Expanded Topics:
{json.dumps(plan.get("expanded_topics", []), ensure_ascii=False)}

Target Venues:
{json.dumps(plan.get("target_venues", []), ensure_ascii=False)}

Target Authors:
{json.dumps(plan.get("target_authors", []), ensure_ascii=False)}

Candidate Papers:
{json.dumps(payload, ensure_ascii=False, indent=2)}

Return ONLY a JSON array. For each paper, include:
{{
  "id": 1,
  "score": 0-10,
  "verdict": "high|medium|low",
  "reason": "one-sentence relevance explanation",
  "matched_aspects": ["aspect"],
  "followup_terms": ["next search term"]
}}

Score by topic fit, methodological relevance, source quality signals, and usefulness for follow-up search."""

    try:
        resp = llm_request([{"role": "user", "content": prompt}], max_tokens=1800, temperature=0.3)
        raw_scores = _safe_json_loads(resp.choices[0].message.content)
        if isinstance(raw_scores, list):
            score_map = {}
            for item in raw_scores:
                try:
                    pid = int(item.get("id"))
                except Exception:
                    continue
                score_map[pid] = item

            scored = []
            for idx, paper in enumerate(candidates, start=1):
                meta = score_map.get(idx, {})
                scored_paper = dict(paper)
                scored_paper["relevance_score"] = min(max(float(meta.get("score", 5.0)), 0.0), 10.0)
                scored_paper["relevance_verdict"] = meta.get("verdict", "medium")
                scored_paper["score_reason"] = meta.get("reason", "")
                scored_paper["matched_aspects"] = meta.get("matched_aspects", [])
                scored_paper["followup_terms"] = meta.get("followup_terms", [])
                scored.append(scored_paper)
            return scored
    except Exception as e:
        print(f"[Warning] Batch scoring failed: {e}, falling back to single-paper scoring")

    scored = []
    for paper in candidates:
        scored_paper = dict(paper)
        scored_paper["relevance_score"] = score_relevance(user_query, paper, plan.get("topic_summary", ""))
        scored_paper["relevance_verdict"] = "medium"
        scored_paper["score_reason"] = ""
        scored_paper["matched_aspects"] = []
        scored_paper["followup_terms"] = []
        scored.append(scored_paper)
    return scored


def refine_query_from_results(original_query: str, papers: list[dict]) -> str:
    """
    Backward-compatible helper used by older flows.
    """
    sample = papers[:5]
    titles = "\n".join([f"- {p.get('title', '')}" for p in sample])
    prompt = f"""Original research query: "{original_query}"

Recent search results (titles):
{titles}

Based on these results, generate a refined search query.
Return ONLY the refined query string."""

    try:
        resp = llm_request([{"role": "user", "content": prompt}], max_tokens=100, temperature=0.7)
        refined = resp.choices[0].message.content.strip().strip('"\'')
        return refined if refined else original_query
    except Exception as e:
        print(f"[Warning] Query refinement failed: {e}")
        return original_query


def refine_search_plan(
    user_query: str,
    initial_plan: dict,
    scored_papers: list[dict],
    max_queries: int = 4,
) -> dict:
    """
    Generate the next round of targeted search queries based on scored results.
    """
    top_papers = sorted(scored_papers, key=lambda x: x.get("relevance_score", 0), reverse=True)[:6]
    payload = [
        {
            "title": paper.get("title", ""),
            "authors": paper.get("authors", ""),
            "year": paper.get("year"),
            "source": paper.get("source", ""),
            "score": paper.get("relevance_score", 0),
            "reason": paper.get("score_reason", ""),
            "matched_aspects": paper.get("matched_aspects", []),
            "followup_terms": paper.get("followup_terms", []),
        }
        for paper in top_papers
    ]

    prompt = f"""You are refining a literature search after reviewing the first-round results.

Original Topic:
{user_query}

Initial Search Plan:
{json.dumps(initial_plan, ensure_ascii=False, indent=2)}

Top Scored Results:
{json.dumps(payload, ensure_ascii=False, indent=2)}

Return ONLY a JSON object with:
{{
  "refinement_summary": "what gaps or promising directions were found",
  "queries": [
    {{
      "query": "next search query",
      "source": "arxiv or ieee",
      "rationale": "why this query should improve recall or precision"
    }}
  ]
}}

Requirements:
1. Use the best papers to infer missing subtopics, methods, venues, or authors.
2. Avoid repeating the exact same queries.
3. Provide at most {max_queries} refined queries."""

    try:
        resp = llm_request([{"role": "user", "content": prompt}], max_tokens=1000, temperature=0.4)
        refined = _safe_json_loads(resp.choices[0].message.content)
        if isinstance(refined, dict) and isinstance(refined.get("queries"), list):
            refined["queries"] = [
                {
                    "query": str(item.get("query", "")).strip(),
                    "source": str(item.get("source", "arxiv")).strip().lower(),
                    "rationale": str(item.get("rationale", "")).strip(),
                }
                for item in refined["queries"]
                if str(item.get("query", "")).strip()
            ][:max_queries]
            if refined["queries"]:
                return refined
    except Exception as e:
        print(f"[Warning] Search refinement failed: {e}, using heuristic refinement")

    heuristic_terms = []
    for paper in top_papers:
        heuristic_terms.extend(paper.get("followup_terms", []))
    heuristic_terms = [term for term in heuristic_terms if term][:max_queries]
    if not heuristic_terms:
        heuristic_terms = [refine_query_from_results(user_query, top_papers)]

    return {
        "refinement_summary": "Fallback refinement based on top-scored results.",
        "queries": [
            {"query": term, "source": "arxiv", "rationale": "Fallback follow-up query"}
            for term in heuristic_terms[:max_queries]
        ],
    }


def multi_round_search(
    user_query: str,
    per_query_limit: int = 8,
    first_round_queries: int = 6,
    second_round_queries: int = 4,
    final_results: int = 10,
) -> dict:
    """
    Multi-round search loop:
    1. LLM expands the topic and creates a search plan.
    2. Execute the first round of retrieval across sources.
    3. LLM scores the candidates and proposes second-round queries.
    4. Execute the refined search and rescore the merged candidates.
    """
    print("\n[Round 1] Planning search strategy...")
    initial_plan = plan_search_strategy(user_query, max_queries=first_round_queries)

    print("[Round 1] Collecting candidates...")
    round1_papers, round1_diagnostics = _collect_papers_from_plan(initial_plan, per_query_limit=per_query_limit)

    if not round1_papers:
        return {
            "query": user_query,
            "error": "No papers found in round 1.",
            "initial_plan": initial_plan,
            "rounds": [{"round": 1, "queries": round1_diagnostics, "papers_found": 0}],
            "results": [],
        }

    print(f"[Round 1] Scoring {len(round1_papers)} candidates...")
    round1_scored = batch_score_papers(user_query, initial_plan, round1_papers, top_k=min(len(round1_papers), 12))

    print("\n[Round 2] Refining search plan...")
    refined_plan = refine_search_plan(
        user_query=user_query,
        initial_plan=initial_plan,
        scored_papers=round1_scored,
        max_queries=second_round_queries,
    )

    print("[Round 2] Collecting refined candidates...")
    round2_papers, round2_diagnostics = _collect_papers_from_plan(
        {"queries": refined_plan.get("queries", [])},
        per_query_limit=per_query_limit,
    )

    merged: dict[str, dict] = {}
    for paper in round1_papers + round2_papers:
        merged[_paper_key(paper)] = dict(paper)

    print(f"[Round 2] Scoring merged set of {len(merged)} candidates...")
    merged_scored = batch_score_papers(
        user_query,
        {
            **initial_plan,
            "refinement_summary": refined_plan.get("refinement_summary", ""),
        },
        list(merged.values()),
        top_k=min(len(merged), 18),
    )
    merged_scored.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)

    return {
        "query": user_query,
        "initial_plan": initial_plan,
        "refined_plan": refined_plan,
        "rounds": [
            {
                "round": 1,
                "queries": round1_diagnostics,
                "papers_found": len(round1_papers),
            },
            {
                "round": 2,
                "queries": round2_diagnostics,
                "papers_found": len(round2_papers),
            },
        ],
        "results": merged_scored[:final_results],
    }


def two_stage_search(
    user_query: str,
    stage1_results: int = 30,
    final_results: int = 10,
    use_multi_query: bool = True,
) -> list[dict]:
    """
    Compatibility wrapper. Older callers expect a flat list of papers.
    """
    del stage1_results, use_multi_query
    result = multi_round_search(
        user_query=user_query,
        per_query_limit=8,
        first_round_queries=6,
        second_round_queries=4,
        final_results=final_results,
    )
    return result.get("results", [])


def iterative_search_with_feedback(
    user_query: str,
    initial_results: int = 10,
    max_iterations: int = 3,
) -> list[dict]:
    """
    Compatibility helper retained for older callers.
    """
    final = multi_round_search(
        user_query=user_query,
        per_query_limit=max(4, initial_results),
        first_round_queries=max_iterations + 2,
        second_round_queries=max_iterations,
        final_results=initial_results * 2,
    )
    return final.get("results", [])

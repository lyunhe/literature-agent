from __future__ import annotations

import json
import os
import re
from html import escape

import networkx as nx
import requests
from pyvis.network import Network
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from . import db
from .config import LIBRARY_DIR, LIBRARY_PDF_DIR
from .llm_client import analyze_pdfs, llm_request
from .paths import display_path, normalize_library_path, pdf_candidates, resolve_library_path


RELATION_COLORS = {
    "cites": "#2f5f9f",
    "cited_by": "#2f5f9f",
    "method_extends": "#cf5f45",
    "method_compares": "#d08a24",
    "same_method_family": "#6f6ab7",
    "same_application": "#3a8f5f",
    "topic_overlap": "#4f8f9f",
    "background_related": "#7b8794",
    "weak_or_unclear": "#c3c8cf",
    "metadata_similarity": "#8a9aa8",
}

OPENALEX_WORKS_URL = "https://api.openalex.org/works"


def _paper_text(paper: dict) -> str:
    return " ".join(
        str(paper.get(field) or "")
        for field in ("title", "abstract", "authors")
    ).strip()


def _node_label(paper: dict) -> str:
    title = paper.get("title") or "Untitled"
    return title if len(title) <= 54 else title[:51] + "..."


def _short(text: str, limit: int = 700) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _node_title(paper: dict) -> str:
    parts = [
        f"<b>{escape(paper.get('title') or 'Untitled')}</b>",
        f"Year: {escape(str(paper.get('year') or ''))}",
        f"Source: {escape(paper.get('source') or '')}",
        f"Authors: {escape(paper.get('authors') or '')}",
    ]
    keywords = paper.get("keywords") or []
    if keywords:
        parts.append(f"Keywords: {escape(', '.join(keywords))}")
    if "degree" in paper:
        parts.append(f"Graph degree: {paper['degree']}")
    if "centrality" in paper:
        parts.append(f"Centrality: {paper['centrality']:.3f}")
    if paper.get("cited_by_count") is not None:
        parts.append(f"OpenAlex cited by: {paper['cited_by_count']}")
    if paper.get("openalex_id"):
        parts.append(f"OpenAlex: {escape(paper['openalex_id'])}")
    if paper.get("pdf_path"):
        parts.append(f"Local PDF: {escape(display_path(paper['pdf_path']))}")
    abstract = paper.get("abstract") or ""
    if abstract:
        parts.append(f"Abstract: {escape(_short(abstract, 700))}")
    return "<br>".join(parts)


def _top_terms(matrix, feature_names, row_index: int, limit: int = 8) -> list[str]:
    row = matrix.getrow(row_index)
    if row.nnz == 0:
        return []
    scored = sorted(
        zip(row.indices, row.data),
        key=lambda item: item[1],
        reverse=True,
    )
    return [feature_names[index] for index, _ in scored[:limit]]


def _shared_keywords(left: list[str], right: list[str], limit: int = 6) -> list[str]:
    left_set = {term.lower(): term for term in left}
    shared = []
    for term in right:
        key = term.lower()
        if key in left_set and key not in {s.lower() for s in shared}:
            shared.append(left_set[key])
        if len(shared) >= limit:
            break
    return shared


def _json_from_text(text: str) -> dict:
    text = (text or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return {}


def _normalize_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (title or "").lower()).strip()


def _resolve_pdf_path(paper: dict) -> str | None:
    for path in pdf_candidates(paper.get("arxiv_id"), paper.get("pdf_path")):
        resolved = resolve_library_path(path)
        if resolved:
            return str(resolved)
    return None


def _fetch_openalex_work(paper: dict, timeout: int = 15) -> dict | None:
    doi = (paper.get("doi") or "").strip()
    if doi:
        doi = doi.removeprefix("https://doi.org/").removeprefix("doi:")
        try:
            resp = requests.get(
                OPENALEX_WORKS_URL,
                params={"filter": f"doi:{doi}", "per-page": 1},
                timeout=timeout,
            )
            resp.raise_for_status()
            results = resp.json().get("results", [])
            if results:
                return results[0]
        except requests.RequestException:
            pass

    title = paper.get("title") or ""
    if not title:
        return None
    try:
        resp = requests.get(
            OPENALEX_WORKS_URL,
            params={"search": title, "per-page": 3},
            timeout=timeout,
        )
        resp.raise_for_status()
        target = _normalize_title(title)
        for candidate in resp.json().get("results", []):
            candidate_title = _normalize_title(candidate.get("title") or "")
            if candidate_title == target or target[:80] in candidate_title:
                return candidate
        results = resp.json().get("results", [])
        return results[0] if results else None
    except requests.RequestException:
        return None


def _load_openalex_metadata(papers: list[dict]) -> dict[str, dict]:
    metadata = {}
    for paper in papers:
        work = _fetch_openalex_work(paper)
        if not work:
            continue
        metadata[str(paper["id"])] = {
            "openalex_id": work.get("id"),
            "doi": work.get("doi"),
            "cited_by_count": work.get("cited_by_count", 0),
            "referenced_works": work.get("referenced_works") or [],
            "publication_year": work.get("publication_year"),
            "type": work.get("type"),
            "openalex_title": work.get("title"),
        }
    return metadata


def _infer_relation_from_pdfs(source: dict, target: dict, relation: dict) -> dict | None:
    source_pdf = _resolve_pdf_path(source)
    target_pdf = _resolve_pdf_path(target)
    if not source_pdf or not target_pdf:
        return None

    prompt = {
        "task": (
            "Read the two PDFs and classify their strongest research relationship. "
            "Focus on method, application, assumptions, experiments, and conclusions. "
            "Do not infer citation unless explicitly visible in the PDFs."
        ),
        "allowed_relation_type": [
            "method_extends",
            "method_compares",
            "same_method_family",
            "same_application",
            "topic_overlap",
            "background_related",
            "weak_or_unclear",
        ],
        "paper_a": {
            "title": source.get("title"),
            "year": source.get("year"),
            "arxiv_id": source.get("arxiv_id"),
        },
        "paper_b": {
            "title": target.get("title"),
            "year": target.get("year"),
            "arxiv_id": target.get("arxiv_id"),
        },
        "metadata_candidate": relation,
        "output_schema": {
            "relation_type": "same_method_family",
            "confidence": 0.0,
            "rationale": "short explanation grounded in the PDFs",
            "evidence_source": "pdf",
        },
    }
    try:
        text = analyze_pdfs(
            [source_pdf, target_pdf],
            "Return strict JSON only:\n" + json.dumps(prompt, ensure_ascii=False),
            max_output_tokens=900,
        )
    except Exception as exc:
        return {
            "pdf_error": str(exc)[:500],
            "source_pdf": source_pdf,
            "target_pdf": target_pdf,
        }
    parsed = _json_from_text(text)
    if not parsed:
        return None
    parsed["source_pdf"] = source_pdf
    parsed["target_pdf"] = target_pdf
    return parsed
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


def _infer_relations_with_llm(
    papers_by_id: dict[str, dict],
    relations: list[dict],
    max_relations: int = 12,
) -> dict[str, dict]:
    """Ask the configured LLM to classify candidate graph edges."""
    candidates = []
    for relation in relations[:max_relations]:
        source_id = str(relation["source_id"])
        target_id = str(relation["target_id"])
        source = papers_by_id[source_id]
        target = papers_by_id[target_id]
        candidates.append({
            "edge_id": f"{source_id}--{target_id}",
            "similarity": relation["weight"],
            "shared_keywords": relation.get("shared_keywords", []),
            "paper_a": {
                "title": source.get("title"),
                "year": source.get("year"),
                "authors": source.get("authors"),
                "abstract": _short(source.get("abstract") or "", 900),
            },
            "paper_b": {
                "title": target.get("title"),
                "year": target.get("year"),
                "authors": target.get("authors"),
                "abstract": _short(target.get("abstract") or "", 900),
            },
        })
    if not candidates:
        return {}

    system = (
        "You classify relationships between academic papers. "
        "Use only the provided titles, abstracts, years, authors, similarity scores, and shared keywords. "
        "Return strict JSON only."
    )
    user = {
        "task": (
            "For each candidate edge, infer the strongest relation type. "
            "Allowed relation_type values: method_extends, method_compares, same_method_family, "
            "same_application, topic_overlap, background_related, weak_or_unclear. "
            "Use method_extends only when one paper plausibly builds on or generalizes the other's method. "
            "Use same_method_family when they share a technique family but extension direction is unclear."
        ),
        "output_schema": {
            "relations": [
                {
                    "edge_id": "source--target",
                    "relation_type": "same_method_family",
                    "confidence": 0.0,
                    "rationale": "short evidence, <= 35 words",
                }
            ]
        },
        "candidates": candidates,
    }
    resp = llm_request(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ],
        tools=None,
        temperature=0,
        max_tokens=1800,
    )
    parsed = _json_from_text(resp.choices[0].message.content)
    rows = parsed.get("relations", []) if isinstance(parsed, dict) else []
    return {
        row.get("edge_id"): row
        for row in rows
        if isinstance(row, dict) and row.get("edge_id")
    }


def build_similarity_graph(
    limit: int = 50,
    threshold: float = 0.18,
    top_k: int = 3,
    infer_llm: bool = False,
    max_llm_relations: int = 12,
    include_citations: bool = False,
    use_pdf: bool = False,
    max_pdf_relations: int = 3,
) -> tuple[nx.Graph, list[dict]]:
    """Build a paper similarity graph from local-library metadata."""
    papers = db.list_papers(limit=limit)
    graph = nx.Graph()
    if not papers:
        return graph, []

    papers_by_id = {str(paper["id"]): paper for paper in papers}
    openalex = _load_openalex_metadata(papers) if include_citations else {}

    for paper in papers:
        paper_id = str(paper["id"])
        openalex_meta = openalex.get(paper_id, {})
        graph.add_node(
            paper_id,
            label=_node_label(paper),
            title=_node_title(paper),
            year=paper.get("year"),
            source=paper.get("source"),
            arxiv_id=paper.get("arxiv_id"),
            doi=paper.get("doi"),
            openalex_id=openalex_meta.get("openalex_id"),
            cited_by_count=openalex_meta.get("cited_by_count"),
            pdf_path=normalize_library_path(_resolve_pdf_path(paper)),
        )

    texts = [_paper_text(paper) for paper in papers]
    if len(papers) < 2 or not any(texts):
        return graph, []

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
    matrix = vectorizer.fit_transform(texts)
    similarities = cosine_similarity(matrix)
    feature_names = vectorizer.get_feature_names_out()
    paper_keywords = {
        str(paper["id"]): _top_terms(matrix, feature_names, index)
        for index, paper in enumerate(papers)
    }

    edges = []
    for i, source in enumerate(papers):
        candidates = []
        for j, target in enumerate(papers):
            if i == j:
                continue
            score = float(similarities[i][j])
            if score >= threshold:
                candidates.append((score, i, j, source, target))
        candidates.sort(reverse=True, key=lambda item: item[0])
        edges.extend(candidates[:top_k])

    seen = set()
    relation_rows = []
    for score, i, j, source, target in sorted(edges, reverse=True, key=lambda item: item[0]):
        source_id = str(source["id"])
        target_id = str(target["id"])
        edge_key = tuple(sorted((source_id, target_id)))
        if edge_key in seen:
            continue
        seen.add(edge_key)
        relation = {
            "source_id": source["id"],
            "target_id": target["id"],
            "source_title": source.get("title"),
            "target_title": target.get("title"),
            "type": "metadata_similarity",
            "weight": round(score, 3),
            "shared_keywords": _shared_keywords(
                paper_keywords.get(source_id, []),
                paper_keywords.get(target_id, []),
            ),
        }
        relation_rows.append(relation)

    llm_rows = {}
    if infer_llm:
        llm_rows = _infer_relations_with_llm(
            papers_by_id=papers_by_id,
            relations=relation_rows,
            max_relations=max_llm_relations,
        )

    if infer_llm and use_pdf:
        pdf_successes = 0
        pdf_attempts = 0
        max_pdf_attempts = max(max_pdf_relations * 3, max_pdf_relations)
        for relation in relation_rows:
            if pdf_successes >= max_pdf_relations or pdf_attempts >= max_pdf_attempts:
                break
            source = papers_by_id[str(relation["source_id"])]
            target = papers_by_id[str(relation["target_id"])]
            pdf_row = _infer_relation_from_pdfs(source, target, relation)
            if not pdf_row:
                continue
            pdf_attempts += 1
            if pdf_row.get("pdf_error"):
                relation["pdf_error"] = pdf_row["pdf_error"]
                relation["source_pdf"] = pdf_row.get("source_pdf")
                relation["target_pdf"] = pdf_row.get("target_pdf")
                continue
            pdf_successes += 1
            relation["type"] = pdf_row.get("relation_type") or relation["type"]
            relation["llm_confidence"] = float(pdf_row.get("confidence") or relation.get("llm_confidence") or 0)
            relation["rationale"] = pdf_row.get("rationale") or relation.get("rationale") or ""
            relation["evidence_source"] = "pdf"
            relation["source_pdf"] = pdf_row.get("source_pdf")
            relation["target_pdf"] = pdf_row.get("target_pdf")

    if include_citations:
        ids_by_openalex = {
            meta["openalex_id"]: paper_id
            for paper_id, meta in openalex.items()
            if meta.get("openalex_id")
        }
        existing_pairs = {
            tuple(sorted((str(row["source_id"]), str(row["target_id"])))): row
            for row in relation_rows
        }
        for source_id, meta in openalex.items():
            referenced = set(meta.get("referenced_works") or [])
            for target_oa_id, target_id in ids_by_openalex.items():
                if source_id == target_id or target_oa_id not in referenced:
                    continue
                pair_key = tuple(sorted((source_id, target_id)))
                source = papers_by_id[source_id]
                target = papers_by_id[target_id]
                citation_row = existing_pairs.get(pair_key)
                if citation_row:
                    citation_row.setdefault("supporting_relations", []).append("cites")
                    citation_row["citation_direction"] = {
                        "citing_id": int(source_id),
                        "cited_id": int(target_id),
                    }
                    if citation_row.get("type") in ("metadata_similarity", "topic_overlap", "background_related", "weak_or_unclear"):
                        citation_row["type"] = "cites"
                    continue
                citation_row = {
                    "source_id": int(source_id),
                    "target_id": int(target_id),
                    "source_title": source.get("title"),
                    "target_title": target.get("title"),
                    "type": "cites",
                    "weight": 1.0,
                    "shared_keywords": [],
                    "citation_direction": {
                        "citing_id": int(source_id),
                        "cited_id": int(target_id),
                    },
                    "rationale": "OpenAlex referenced_works shows a citation between these local-library papers.",
                    "evidence_source": "openalex",
                }
                relation_rows.append(citation_row)
                existing_pairs[pair_key] = citation_row

    for relation in relation_rows:
        source_id = str(relation["source_id"])
        target_id = str(relation["target_id"])
        edge_id = f"{source_id}--{target_id}"
        llm_relation = llm_rows.get(edge_id)
        if llm_relation and relation.get("evidence_source") != "pdf":
            relation["type"] = llm_relation.get("relation_type") or relation["type"]
            relation["llm_confidence"] = float(llm_relation.get("confidence") or 0)
            relation["rationale"] = llm_relation.get("rationale") or ""
        title_parts = [
            f"Relation: {escape(relation['type'])}",
            f"Similarity: {relation['weight']:.3f}",
        ]
        if relation.get("shared_keywords"):
            title_parts.append(f"Shared keywords: {escape(', '.join(relation['shared_keywords']))}")
        if relation.get("rationale"):
            title_parts.append(f"LLM rationale: {escape(relation['rationale'])}")
        if relation.get("evidence_source"):
            title_parts.append(f"Evidence source: {escape(relation['evidence_source'])}")
        if relation.get("pdf_error"):
            title_parts.append(f"PDF read error: {escape(relation['pdf_error'])}")
        if relation.get("supporting_relations"):
            title_parts.append(f"Supporting relations: {escape(', '.join(relation['supporting_relations']))}")
        if relation.get("citation_direction"):
            direction = relation["citation_direction"]
            title_parts.append(f"Citation direction: {direction['citing_id']} cites {direction['cited_id']}")
        if relation.get("llm_confidence") is not None:
            title_parts.append(f"LLM confidence: {relation['llm_confidence']:.2f}")
        graph.add_edge(
            source_id,
            target_id,
            weight=relation["weight"],
            title="<br>".join(title_parts),
            label=f"{relation['type']}\n{relation['weight']:.2f}",
            relation=relation["type"],
            shared_keywords=relation.get("shared_keywords", []),
            llm_confidence=relation.get("llm_confidence"),
            rationale=relation.get("rationale"),
            evidence_source=relation.get("evidence_source"),
            citation_direction=relation.get("citation_direction"),
        )

    centrality = nx.degree_centrality(graph) if graph.number_of_nodes() else {}
    for paper in papers:
        node_id = str(paper["id"])
        graph.nodes[node_id]["keywords"] = paper_keywords.get(node_id, [])
        graph.nodes[node_id]["degree"] = graph.degree[node_id]
        graph.nodes[node_id]["centrality"] = round(centrality.get(node_id, 0), 3)
        paper_for_title = {
            **paper,
            "keywords": graph.nodes[node_id]["keywords"],
            "degree": graph.nodes[node_id]["degree"],
            "centrality": graph.nodes[node_id]["centrality"],
            "openalex_id": graph.nodes[node_id].get("openalex_id"),
            "cited_by_count": graph.nodes[node_id].get("cited_by_count"),
            "pdf_path": graph.nodes[node_id].get("pdf_path"),
        }
        graph.nodes[node_id]["title"] = _node_title(paper_for_title)

    return graph, relation_rows


def export_similarity_graph(
    output_path: str | None = None,
    limit: int = 50,
    threshold: float = 0.18,
    top_k: int = 3,
    infer_llm: bool = False,
    max_llm_relations: int = 12,
    include_citations: bool = False,
    use_pdf: bool = False,
    max_pdf_relations: int = 3,
) -> dict:
    """Export a local-library similarity graph to HTML and JSON."""
    graph, relations = build_similarity_graph(
        limit=limit,
        threshold=threshold,
        top_k=top_k,
        infer_llm=infer_llm,
        max_llm_relations=max_llm_relations,
        include_citations=include_citations,
        use_pdf=use_pdf,
        max_pdf_relations=max_pdf_relations,
    )
    graph_dir = os.path.join(LIBRARY_DIR, "graphs")
    os.makedirs(graph_dir, exist_ok=True)
    if output_path is None:
        output_path = os.path.join(graph_dir, "literature_network.html")
    json_path = os.path.splitext(output_path)[0] + ".json"

    net = Network(
        height="760px",
        width="100%",
        bgcolor="#ffffff",
        font_color="#1f2937",
        notebook=False,
        cdn_resources="in_line",
    )
    net.barnes_hut(gravity=-4200, central_gravity=0.18, spring_length=180)

    for node_id, attrs in graph.nodes(data=True):
        value = 12 + attrs.get("degree", 0) * 5
        net.add_node(
            node_id,
            label=attrs.get("label"),
            title=attrs.get("title"),
            value=value,
            color="#4f8f9f",
        )
    for source, target, attrs in graph.edges(data=True):
        width = 1 + attrs.get("weight", 0) * 8
        relation = attrs.get("relation", "metadata_similarity")
        net.add_edge(
            source,
            target,
            value=attrs.get("weight", 0),
            width=width,
            title=attrs.get("title"),
            label=attrs.get("label"),
            color=RELATION_COLORS.get(relation, "#8a9aa8"),
        )

    net.show_buttons(filter_=["physics"])
    net.write_html(output_path, open_browser=False, notebook=False)

    summary = {
        "html_path": os.path.abspath(output_path),
        "json_path": os.path.abspath(json_path),
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "llm_inference": infer_llm,
        "citation_enrichment": include_citations,
        "pdf_inference": use_pdf,
        "relation_types": sorted({rel.get("type") for rel in relations if rel.get("type")}),
        "node_metrics": [
            {
                "id": node_id,
                "label": attrs.get("label"),
                "degree": attrs.get("degree", 0),
                "centrality": attrs.get("centrality", 0),
                "keywords": attrs.get("keywords", []),
                "openalex_id": attrs.get("openalex_id"),
                "cited_by_count": attrs.get("cited_by_count"),
                "pdf_path": attrs.get("pdf_path"),
                "pdf_abs_path": display_path(attrs.get("pdf_path")),
            }
            for node_id, attrs in graph.nodes(data=True)
        ],
        "relations": relations,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary

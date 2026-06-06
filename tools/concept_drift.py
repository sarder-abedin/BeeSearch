"""
tools/concept_drift.py
──────────────────────
Detects vocabulary evolution in a paper corpus over time.

Algorithm:
  1. Group papers into 5-year buckets
  2. Extract TF-IDF keywords per bucket (stdlib only — no scikit-learn dependency)
  3. Track keyword rank changes across buckets
  4. Classify terms as rising, declining, or stable
  5. Optionally use LLM to narrate the conceptual shifts
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "this", "that", "these",
    "those", "it", "its", "we", "our", "their", "they", "study", "research",
    "paper", "results", "using", "used", "based", "analysis", "data",
    "method", "methods", "review", "systematic", "meta", "found", "show",
    "showed", "evidence", "significant", "effect", "effects", "associated",
    "patients", "participants", "reported", "compared", "between", "also",
    "than", "more", "less", "both", "can", "two", "one", "three",
}


def _tokenize(text: str) -> List[str]:
    return [w for w in re.findall(r"\b[a-z]{3,}\b", text.lower()) if w not in _STOPWORDS]


def _tfidf_top(texts: List[str], top_n: int = 15) -> List[str]:
    """TF-IDF keyword extraction from a list of texts (no external deps)."""
    if not texts:
        return []
    doc_tokens = [_tokenize(t) for t in texts]
    n_docs = len(doc_tokens)
    doc_freq: Counter = Counter()
    for tokens in doc_tokens:
        doc_freq.update(set(tokens))
    idf = {t: math.log((n_docs + 1) / (f + 1)) + 1 for t, f in doc_freq.items()}
    scores: Counter = Counter()
    for tf_tokens in doc_tokens:
        total = len(tf_tokens) or 1
        for term, cnt in Counter(tf_tokens).items():
            scores[term] += (cnt / total) * idf.get(term, 1.0)
    return [t for t, _ in scores.most_common(top_n)]


def _bucket_papers(papers: List[Dict], bucket_size: int = 5) -> Dict[str, List[Dict]]:
    """Group papers into consecutive year buckets of `bucket_size` years."""
    buckets: Dict[str, List[Dict]] = defaultdict(list)
    for p in papers:
        year = p.get("year")
        try:
            y = int(year)
            start = (y // bucket_size) * bucket_size
            label = f"{start}–{start + bucket_size - 1}"
            buckets[label].append(p)
        except (TypeError, ValueError):
            buckets["Unknown"].append(p)
    return dict(sorted(buckets.items()))


def detect_concept_drift(
    papers: List[Dict],
    model_name: str = "llama3.1:8b",
    num_ctx: int = 32768,
    bucket_size: int = 5,
    top_n: int = 12,
) -> Dict[str, Any]:
    """
    Analyze how the field's vocabulary has shifted across time buckets.

    Returns
    -------
    {
      buckets        : {label: {"papers": N, "top_terms": [...]}}
      rising_terms   : [{"term", "first_bucket", "last_bucket", "growth", "scores"}]
      declining_terms: [...same structure...]
      stable_terms   : [...same structure...]
      llm_analysis   : prose narrative from LLM (empty string if LLM call fails)
    }
    """
    if not papers:
        return {"buckets": {}, "rising_terms": [], "declining_terms": [], "stable_terms": [], "llm_analysis": ""}

    buckets = _bucket_papers(papers, bucket_size)
    bucket_kw: Dict[str, List[str]] = {}
    bucket_meta: Dict[str, Dict] = {}

    for label, plist in buckets.items():
        if label == "Unknown":
            continue
        texts = [(p.get("title", "") + " " + p.get("abstract", "")) for p in plist]
        kw = _tfidf_top(texts, top_n)
        bucket_kw[label] = kw
        bucket_meta[label] = {"papers": len(plist), "top_terms": kw}

    bucket_labels = sorted(bucket_kw.keys())
    term_rank: Dict[str, Dict[str, int]] = defaultdict(dict)
    for label in bucket_labels:
        for rank, term in enumerate(bucket_kw.get(label, [])):
            term_rank[term][label] = top_n - rank  # higher score = more prominent

    rising, declining, stable = [], [], []

    for term, presence in term_rank.items():
        labels_present = sorted(presence.keys())
        if len(labels_present) < 2:
            continue
        first_score = presence[labels_present[0]]
        last_score = presence[labels_present[-1]]
        growth = last_score - first_score
        entry = {
            "term": term,
            "first_bucket": labels_present[0],
            "last_bucket": labels_present[-1],
            "n_buckets": len(labels_present),
            "growth": growth,
            "scores": {b: presence.get(b, 0) for b in bucket_labels},
        }
        if growth >= 3:
            rising.append(entry)
        elif growth <= -3:
            declining.append(entry)
        else:
            stable.append(entry)

    rising.sort(key=lambda x: -x["growth"])
    declining.sort(key=lambda x: x["growth"])

    llm_analysis = ""
    if bucket_meta and (rising or declining):
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            from langchain_ollama import ChatOllama
            from config.settings import get_settings
            import httpx
            cfg_s = get_settings()
            llm = ChatOllama(
                model=model_name or cfg_s.ollama_model,
                base_url=cfg_s.ollama_base_url,
                temperature=0.3,
                num_predict=512,
                num_ctx=num_ctx,
                sync_client_kwargs={"timeout": httpx.Timeout(120.0)},
            )
            period_lines = [
                f"{label} ({meta['papers']} papers): {', '.join(meta['top_terms'][:8])}"
                for label, meta in list(bucket_meta.items())[:5]
            ]
            rising_str = ", ".join(r["term"] for r in rising[:5])
            declining_str = ", ".join(d["term"] for d in declining[:5])
            response = llm.invoke([
                SystemMessage(content=(
                    "You are a research methodology expert. Analyze vocabulary evolution in a research corpus. "
                    "Identify conceptual shifts, emerging paradigms, and declining frameworks. "
                    "Write 2-3 paragraphs. Be specific about what terminology shifts reveal about the field."
                )),
                HumanMessage(content=(
                    "Vocabulary by time period:\n" + "\n".join(period_lines) + "\n\n"
                    f"Rising terms (newer prominence): {rising_str}\n"
                    f"Declining terms (older prominence): {declining_str}"
                )),
            ])
            llm_analysis = response.content.strip()
        except Exception as e:
            logger.warning("Concept drift LLM analysis failed: %s", e)

    return {
        "buckets": bucket_meta,
        "rising_terms": rising[:10],
        "declining_terms": declining[:10],
        "stable_terms": stable[:10],
        "llm_analysis": llm_analysis,
    }

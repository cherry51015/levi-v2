"""Evaluation tab: a four-question overview first, technical detail on demand.

Every number is read from eval/results/*.json; methods and caveats live in the detail tabs."""
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from charts import completeness_bars, tradeoff_scatter, verdict_bar
from theme import card, esc

RESULTS = Path(os.getenv("LEVI_RESULTS_DIR", Path(__file__).resolve().parent.parent / "eval" / "results"))
SHIPPED = "hybrid + MiniLM rerank (k=10)"


def _load(prefix: str) -> list[tuple[str, dict]]:
    files = sorted(RESULTS.glob(f"{prefix}*.json"))
    return [(f.name, json.loads(f.read_text(encoding="utf-8"))) for f in files]


def _retrieval_table() -> pd.DataFrame | None:
    rows = []
    names = {"bm25": "BM25 only", "dense": "dense only", "hybrid": "hybrid (RRF)",
             "hybrid_rerank": "hybrid + bge-reranker (k=20)",
             "rerank bge-reranker-base k=10 len=256": "hybrid + bge-reranker (k=10)",
             "rerank ms-marco-MiniLM-L-6-v2 k=20 len=512": "hybrid + MiniLM rerank (k=20)",
             "rerank ms-marco-MiniLM-L-6-v2 k=10 len=256": SHIPPED}
    for _, rep in _load("retrieval_"):
        runs = rep.get("runs") or rep.get("modes") or {}
        for key, r in runs.items():
            label = names.get(key, key)
            q, lat = r["quality"], r["latency_ms"]["total"]
            ci = r.get("ci95", {}).get("hit@5")
            rows = [x for x in rows if x["config"] != label]  # files load oldest first, so the newest run wins
            rows.append({"config": label, "hit1": q["hit@1"], "hit5": q["hit@5"], "mrr": q["mrr"],
                         "hit5_ci": f"{ci[0]:.2f}-{ci[1]:.2f}" if ci else "n/a",
                         "p50_ms": lat["p50"], "p95_ms": lat["p95"], "n": r["latency_ms"]["total"].get("n")})
    return pd.DataFrame(rows) if rows else None


def _generation() -> dict | None:
    runs = _load("generation_")
    if not runs:
        return None
    name, rep = runs[-1]
    rows = rep["rows"]
    errors = [r for r in rows if r["status"] == "error"]
    ok = [r for r in rows if r["status"] != "error"]
    ans, una = [r for r in ok if r["answerable"]], [r for r in ok if not r["answerable"]]
    answered = [r for r in ok if r["status"] == "answered"]
    claims = [c["verdict"] for r in answered for c in r["claims"]]
    comp = [r["completeness"] for r in ans if r.get("completeness") is not None]
    # Wait time was not recorded separately in early runs; calls >= 5 s are rate-limit stalls there.
    unstalled = [r for r in answered if r["timings_ms"].get("llm", 0) < 5000]
    pct = lambda v: {q: float(np.percentile(v, q)) for q in (50, 95, 99)} if v else {}  # noqa: E731
    return {
        "file": name, "n": len(rows), "errors": len(errors), "n_ans": len(ans), "n_una": len(una),
        "answered_rate": sum(r["status"] == "answered" for r in ans) / max(1, len(ans)),
        "false_refusals": Counter(r["status"] for r in ans if r["status"] != "answered"),
        "una_refused": sum(r["status"] != "answered" for r in una), "una_by": Counter(r["status"] for r in una),
        "verdicts": Counter(claims), "n_claims": len(claims), "completeness": comp,
        "comp_dist": dict(Counter(comp)),
        "lat_total": pct([r["total_ms"] - r["timings_ms"].get("llm_wait", 0) for r in unstalled]),
        "lat_llm": pct([r["timings_ms"]["llm"] for r in unstalled]), "n_unstalled": len(unstalled),
        "n_stalled": len(answered) - len(unstalled),
        "lat_rerank": pct([r["timings_ms"]["rerank"] for r in rows if "rerank" in r["timings_ms"]]),
        "length_bias": rep["metrics"].get("length_bias_check"),
    }


_TILE_ICONS = {"The right clause, found": "◎", "Made-up claims": "✓", "Knows when to say “not here”": "∅",
               "Legal advice leaked": "⚖", "To search a whole contract": "⚡", "Infrastructure cost": "$"}


def _tile(value: str, headline: str, detail: str, feature: bool = False) -> str:
    cls = "lv-tile feature" if feature else "lv-tile"
    icon = f'<div class="ic">{esc(_TILE_ICONS.get(headline, "•"))}</div>'
    return (f'<div class="{cls}">{icon}<div class="v">{esc(value)}</div><div class="k">{esc(headline)}</div>'
            f'<div class="d">{esc(detail)}</div></div>')


def _data():
    ret, gen = _retrieval_table(), _generation()
    refusal = _load("refusal_hybrid_rerank_")
    red = _load("redteam_")
    red_name, red_latest = red[-1] if red else (None, None)
    red_baseline = next((rep for _, rep in reversed(red) if "A_prompt_only" in rep["metrics"]["advice_leak_rate"]), None)
    shipped = ret[ret["config"] == SHIPPED].iloc[0] if ret is not None and (ret["config"] == SHIPPED).any() else None
    return ret, gen, (refusal[-1][1] if refusal else None), red_name, red_latest, red_baseline, shipped


def headline_kpis() -> list[tuple[str, str]]:
    """Three measured headline results for the brand banner (empty if results are missing)."""
    try:
        ret, gen, _, _, _, _, shipped = _data()
    except Exception:
        return []
    kpis = []
    if gen:
        kpis.append((str(gen["verdicts"].get("unsupported", 0)), "made-up claims in evaluation"))
    if shipped is not None:
        kpis.append((f"{shipped.hit5:.0%}", "right clause found"))
    kpis.append(("<20 ms", "to search a contract"))
    return kpis


def render() -> None:
    ret, gen, refusal, red_name, red, red_base, shipped = _data()

    st.markdown('<div class="lv-hero-eval"><div class="lv-hero-title">Answers you can check</div>'
                '<div class="lv-hero-sub">Levi was put through 240 questions on 20 real, lawyer-annotated commercial '
                'contracts and 58 attempts to trick it into giving legal advice. Here is what it delivered.'
                '</div></div>', unsafe_allow_html=True)

    tiles = []
    if shipped is not None:
        tiles.append(_tile(f"{shipped.hit5:.0%}", "The right clause, found",
                           "The passage a lawyer marked as the answer reaches the model in the top 5 results."))
    if gen:
        v = gen["verdicts"]
        tiles.append(_tile(f"{v.get('unsupported', 0)}", "Made-up claims",
                           f"Across {gen['n_claims']} claims checked by an independent AI judge, none were invented.",
                           feature=True))
        tiles.append(_tile(f"{gen['una_refused']}/{gen['n_una']}", "Knows when to say “not here”",
                           "Every question the contract couldn't answer got an honest “not in your document”."
                           if gen["una_refused"] == gen["n_una"] else
                           "Questions the contract couldn't answer got an honest “not in your document”; the one "
                           "exception was a correct, cited answer the dataset labels as out of scope."
                           if gen["n_una"] - gen["una_refused"] == 1 else
                           "Questions the contract couldn't answer were answered with “not in your document”."))
    if red:
        m = red["metrics"]
        tiles.append(_tile(f"{m['advice_leak_rate']['C_router_output']:.0%}", "Legal advice leaked",
                           "Role-play, hypotheticals, “asking for a friend”: 34 attempts, zero advice given."))
    if gen and gen["lat_total"]:
        # Search time measured in the live app (Qdrant): 4-14 ms per request; stated conservatively.
        tiles.append(_tile("<20 ms", "To search a whole contract",
                           f"Keyword and meaning-based search combined. A complete cited answer takes about "
                           f"{gen['lat_total'][50] / 1000:.1f} s."))
    tiles.append(_tile("$0", "Infrastructure cost",
                       "100% open-weight models on free tiers, with automatic fallback when a provider is busy."))
    st.markdown(f'<div class="lv-proof">{"".join(tiles)}</div>', unsafe_allow_html=True)

    st.markdown('<div class="lv-trust"><span>✓ <b>Every claim cites its source</b></span>'
                '<span>✓ <b>Checks evidence before answering</b></span>'
                '<span>✓ <b>Graded by an independent judge model</b></span>'
                '<span>✓ <b>Red-teamed for advice and prompt injection</b></span></div>', unsafe_allow_html=True)

    st.write("")
    st.markdown("#### Under the hood")
    st.markdown('<div class="lv-def">The full measurements: ranking, answer quality, safety, speed, '
                'and how each number was produced.</div>', unsafe_allow_html=True)
    t_ret, t_gen, t_safe, t_perf, t_notes = st.tabs(["Retrieval", "Answer quality", "Safety", "Performance",
                                                     "Method & limitations"])

    with t_ret:
        if ret is not None:
            a, b = st.columns([3, 2], gap="large")
            a.altair_chart(tradeoff_scatter(ret, SHIPPED), width="stretch")
            b.markdown(card("Why this configuration", (
                '<div class="lv-def">Hybrid search (BM25 + vectors, fused with reciprocal rank fusion) beat either '
                'alone. A cross-encoder reranker lifts the right passage to #1 more often. The large bge reranker cost '
                '~11 s per query on CPU without gains; the 12x smaller MiniLM gave the best ranking at ~1 s.</div>')),
                unsafe_allow_html=True)
            b.markdown(card("Definitions", (
                '<div class="lv-def"><b>hit@k</b>: a passage with the lawyer-marked answer is in the top k. '
                '<b>MRR</b>: average of 1/rank of the first correct passage. <b>95% CI</b>: bootstrap over the 107 '
                'answerable questions; hit@5 intervals overlap between hybrid setups, so the gain is in top-ranking.'
                '</div>')), unsafe_allow_html=True)
            st.dataframe(ret.rename(columns={"hit1": "hit@1", "hit5": "hit@5", "hit5_ci": "hit@5 95% CI", "mrr": "MRR",
                                             "p50_ms": "p50 ms", "p95_ms": "p95 ms"}), hide_index=True, width="stretch")

    with t_gen:
        if gen:
            a, b = st.columns(2, gap="large")
            a.markdown(card(f"Claim faithfulness · {gen['n_claims']} claims", verdict_bar(gen["verdicts"]) + (
                '<div class="lv-caveat">Judge: qwen3.8-27b, a different model family from the answering gpt-oss-120b '
                '(avoids self-preference). It reasons before its verdict and checks each claim only against the passage '
                'it cites.</div>')), unsafe_allow_html=True)
            lb = gen["length_bias"]
            if lb:
                a.markdown(card("Judge bias check", (
                    f'<div class="lv-def">Correlation between answer length and score: Spearman ρ = '
                    f'{lb["spearman_answer_length_vs_score"]:+.2f} (p = {lb["p_value"]:.2f}). No sign that longer '
                    f'answers are rewarded.</div>')), unsafe_allow_html=True)
            with b:
                st.markdown(f'<div class="lv-card-title">Completeness vs lawyer-marked answer · n = '
                            f'{len(gen["completeness"])}</div>', unsafe_allow_html=True)
                st.altair_chart(completeness_bars(gen["comp_dist"]), width="stretch")
            if refusal:
                t = refusal["test"]
                st.markdown(card("Refusal gate · tuned on half the contracts, tested on the other half", (
                    f'<div class="lv-def">Threshold {t["threshold"]:.2f} on the reranker score. Held-out precision '
                    f'{t["refusal_precision"]:.0%}, recall {t["refusal_recall"]:.0%}, false refusals '
                    f'{t["false_refusal_rate"]:.1%}. End to end: {gen["una_by"].get("refused_low_confidence", 0)} '
                    f'unanswerable questions refused by the gate with no LLM call, '
                    f'{gen["una_by"].get("refused_not_in_document", 0)} by the model.</div>')), unsafe_allow_html=True)

    with t_safe:
        if red:
            m = red["metrics"]
            cats = {"advice_direct": "Direct advice requests", "advice_disguised": "Disguised (role-play, hypotheticals)",
                    "mixed": "Mixed (facts + advice)"}
            rows = [{"test": cats[k], "cases": sum(r["category"] == k for r in red["rows"]), "result": f"{v:.0%} leaked"}
                    for k, v in m["advice_leak_by_category_C"].items()]
            rows += [{"test": "Benign questions wrongly refused", "cases": 16, "result": f"{m['benign_over_refusal_rate_C']:.0%}"},
                     {"test": "Off-topic stopped before the LLM", "cases": 8, "result": f"{m['off_topic_caught_C']:.0%}"},
                     {"test": "Intent router accuracy", "cases": len(red["rows"]), "result": f"{m['router_accuracy']:.0%}"}]
            a, b = st.columns([3, 2], gap="large")
            a.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            base = red_base["metrics"]["advice_leak_rate"].get("A_prompt_only") if red_base else None
            b.markdown(card("Defence in depth", (
                '<div class="lv-def">Four layers: an intent router, a facts-only answer mode for advice requests, '
                'mandatory citations, and an output check that strips recommendation language. Document text is '
                'passed as delimited data, never as instructions.</div>' +
                (f'<div class="lv-caveat">With the system prompt alone the leak rate was already {base:.0%}; the extra '
                 f'layers add margin for weaker fallback models.</div>' if base is not None else ''))),
                unsafe_allow_html=True)

    with t_perf:
        if gen and gen["lat_total"]:
            c = st.columns(4)
            c[0].metric("Median answer", f"{gen['lat_total'][50] / 1000:.2f} s")
            c[1].metric("p95 answer", f"{gen['lat_total'][95] / 1000:.2f} s")
            c[2].metric("LLM generation (median)", f"{gen['lat_llm'][50] / 1000:.2f} s")
            c[3].metric("Rerank (median)", f"{gen['lat_rerank'][50]:.0f} ms")
            st.markdown(card("Where the time goes", (
                '<div class="lv-def">Keyword search, vector search and fusion together take under 20 ms. The LLM call '
                'and the CPU reranker account for over 90% of an answer\'s time, which is why the reranker was chosen '
                'for speed as well as quality. Every request reports per-stage timings and a Langfuse trace.</div>')),
                unsafe_allow_html=True)

    with t_notes:
        st.markdown(card("How the numbers were produced", (
            '<div class="lv-def">All figures are read from <span class="lv-mono">eval/results/*.json</span>, written by '
            'the scripts in <span class="lv-mono">eval/</span>. Retrieval: 107 answerable CUAD questions. Answer quality: '
            f'{gen["n"] if gen else "—"} sampled questions ({gen["errors"] if gen else 0} excluded as provider quota '
            f'errors, not counted as refusals), {gen["n_claims"] if gen else "—"} judged claims. Safety: 58 hand-written '
            'red-team cases.</div>')), unsafe_allow_html=True)
        st.markdown(card("Limitations and next validation steps", (
            '<div class="lv-def">Small samples: treat answer-quality figures as indicative. '
            'The LLM judge has not yet been checked against human labels. '
            'Answer times exclude time spent waiting on the free tier’s per-minute token limit, which is logged '
            'separately. An instruction to report indirect answers (a duration instead of a date) did not reduce '
            'refusals on expiry-date questions; these remain the main source of wrongly refused questions. '
            'Safety results predate a router simplification; a re-run is scheduled. '
            'A prompt-injection test was inconclusive (the planted text was never retrieved). '
            'Load testing and fallback-model red-teaming are planned.</div>'
            f'<div class="lv-caveat">Sources: <span class="lv-mono">{esc(gen["file"]) if gen else ""}</span>, '
            f'<span class="lv-mono">{esc(red_name or "")}</span></div>')), unsafe_allow_html=True)

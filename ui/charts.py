"""Charts for the inspector and the evaluation dashboard.

Rules followed: one series -> one colour; emphasis (accent vs muted gray) instead of
many hues; thin marks with 4px rounded ends; hairline recessive axes; selective direct
labels; a tooltip on every mark; status colours only with icon + label.
"""
import altair as alt
import pandas as pd

from theme import (ACCENT, CRITICAL, GOOD, HAIRLINE, INK, INK_2, MUTED, STAGE_NAMES, STAGE_ORDER, WAIT_GRAY, WARNING,
                   esc)

FONT = "system-ui, -apple-system, Segoe UI, sans-serif"


def _style(chart: alt.Chart) -> alt.Chart:
    return (chart.configure_view(strokeWidth=0)
            .configure_axis(labelColor=INK_2, titleColor=INK_2, gridColor=HAIRLINE, domainColor="#c3c2b7",
                            tickColor="#c3c2b7", labelFont=FONT, titleFont=FONT, labelFontSize=11, titleFontSize=11,
                            titleFontWeight="normal")
            .configure_text(font=FONT)
            .configure(background="transparent"))


def stage_timeline(timings: dict[str, float]) -> alt.Chart:
    """Waterfall of pipeline stages: where did this answer's time go?"""
    order = [s for s in STAGE_ORDER if s in timings] + [s for s in timings if s not in STAGE_ORDER]
    rows, t = [], 0.0
    total = sum(timings.values()) or 1.0
    for s in order:
        ms = timings[s]
        rows.append({"stage": STAGE_NAMES.get(s, s), "start": t, "end": t + max(ms, total * 0.004), "ms": round(ms, 1),
                     "share": f"{ms / total:.0%}", "kind": "wait" if s == "llm_wait" else "work",
                     "label": (f"{ms / 1000:.2f} s" if ms >= 1000 else f"{ms:.0f} ms") if ms >= total * 0.06 else ""})
        t += ms
    df = pd.DataFrame(rows)
    y = alt.Y("stage:N", sort=list(df["stage"]), title=None,
              axis=alt.Axis(grid=False, ticks=False, domain=False, labelLimit=180, labelOverlap=False, labelPadding=8))
    bars = alt.Chart(df).mark_bar(height=12, cornerRadius=4).encode(
        x=alt.X("start:Q", title="milliseconds since request start", axis=alt.Axis(tickCount=5)),
        x2="end:Q", y=y,
        color=alt.Color("kind:N", scale=alt.Scale(domain=["work", "wait"], range=[ACCENT, WAIT_GRAY]), legend=None),
        tooltip=[alt.Tooltip("stage:N"), alt.Tooltip("ms:Q", title="duration (ms)"), alt.Tooltip("share:N", title="share")],
    )
    labels = alt.Chart(df).mark_text(align="left", dx=5, fontSize=11, color=INK_2).encode(x="end:Q", y=y, text="label:N")
    return _style((bars + labels).properties(height=max(140, 30 * len(df)), width="container"))


def confidence_track(score: float | None, threshold: float, lo: float = -12.0, hi: float = 10.0) -> str:
    """Retrieval confidence on the reranker's logit scale, with the refusal cut-off marked."""
    if score is None:
        return '<div class="lv-def">No retrieval ran for this question.</div>'
    pct = lambda v: max(0.0, min(100.0, (v - lo) / (hi - lo) * 100))  # noqa: E731
    passed = score >= threshold
    colour, verdict = (GOOD, "✓ above cut-off: sent to the LLM") if passed else (MUTED, "∅ below cut-off: refused without an LLM call")
    return (f'<div class="lv-track"><div class="fill" style="width:{pct(score):.1f}%;background:{ACCENT}"></div>'
            f'<div class="cut" style="left:{pct(threshold):.1f}%"></div>'
            f'<div class="cutlabel" style="left:{pct(threshold):.1f}%">refusal cut-off {threshold:g}</div></div>'
            f'<div class="lv-scale"><span>{lo:g} (irrelevant)</span><span>{hi:g} (strong match)</span></div>'
            f'<div class="lv-def" style="margin-top:6px"><b style="color:{INK}">{score:.2f}</b> · '
            f'<span style="color:{colour if passed else INK_2}">{esc(verdict)}</span></div>')


def verdict_bar(counts: dict[str, int]) -> str:
    """Claim-support verdicts as one segmented bar (part-to-whole, 3 segments, status colours + labels)."""
    total = sum(counts.values()) or 1
    parts = [("supported", GOOD, "✓ supported"), ("partial", WARNING, "◐ partially supported"),
             ("unsupported", CRITICAL, "✕ unsupported")]
    segs = "".join(f'<div title="{lab}: {counts.get(k, 0)}" style="flex:{counts.get(k, 0)};background:{c}"></div>'
                   for k, c, lab in parts if counts.get(k, 0))
    legend = "".join(f'<span><span class="lv-dot" style="background:{c}"></span> {lab} <b style="color:{INK}">'
                     f'{counts.get(k, 0)}</b> ({counts.get(k, 0) / total:.0%})</span>' for k, c, lab in parts)
    return f'<div class="lv-seg">{segs}</div><div class="lv-legend">{legend}</div>'


def completeness_bars(distribution: dict) -> alt.Chart:
    df = pd.DataFrame([{"score": int(k), "answers": v} for k, v in distribution.items()])
    df = df.set_index("score").reindex(range(1, 6), fill_value=0).reset_index()
    base = alt.Chart(df).encode(x=alt.X("score:O", title="judge score (1 = wrong, 5 = complete)",
                                        axis=alt.Axis(labelAngle=0, grid=False)))
    bars = base.mark_bar(size=26, cornerRadiusTopLeft=4, cornerRadiusTopRight=4, color=ACCENT).encode(
        y=alt.Y("answers:Q", title="answers", axis=alt.Axis(tickMinStep=1, grid=True)),
        tooltip=[alt.Tooltip("score:O"), alt.Tooltip("answers:Q")])
    text = base.mark_text(dy=-7, fontSize=11, color=INK_2).encode(y="answers:Q", text="answers:Q")
    return _style((bars + text).properties(height=190, width="container"))


def tradeoff_scatter(df: pd.DataFrame, shipped: str) -> alt.Chart:
    """Retrieval quality vs latency per configuration. One axis per quantity: y = MRR, x = p50 (log)."""
    df = df.assign(role=["shipped" if c == shipped else "tested" for c in df["config"]])
    enc = dict(
        x=alt.X("p50_ms:Q", scale=alt.Scale(type="log"), title="median latency per query (ms, log scale)",
                  axis=alt.Axis(values=[1, 10, 100, 1000, 10000], format=",")),
        y=alt.Y("mrr:Q", scale=alt.Scale(zero=False, padding=12), title="MRR (higher = right chunk ranked higher)"),
    )
    points = alt.Chart(df).mark_circle(size=110, opacity=1, stroke="#fcfcfb", strokeWidth=2).encode(
        **enc, color=alt.Color("role:N", scale=alt.Scale(domain=["shipped", "tested"], range=[ACCENT, MUTED]),
                               legend=alt.Legend(title=None, orient="bottom")),
        tooltip=[alt.Tooltip("config:N"), alt.Tooltip("hit1:Q", title="hit@1", format=".3f"),
                 alt.Tooltip("hit5:Q", title="hit@5", format=".3f"), alt.Tooltip("hit5_ci:N", title="hit@5 95% CI"),
                 alt.Tooltip("mrr:Q", title="MRR", format=".3f"), alt.Tooltip("p50_ms:Q", title="p50 ms", format=",.0f"),
                 alt.Tooltip("p95_ms:Q", title="p95 ms", format=",.0f")])
    labels = alt.Chart(df).mark_text(align="left", dx=9, dy=-8, fontSize=11, color=INK_2).encode(**enc, text="config:N")
    return _style((points + labels).properties(height=320, width="container"))

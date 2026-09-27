"""Visual system for the Levi UI: colour tokens, CSS, and small HTML building blocks.

Palette follows a validated data-viz reference palette: one accent (blue) for
identity, a fixed status palette (good / warning / critical) that always ships
with an icon + text label, and hairline chrome so the content carries the page.
"""
import html
import re

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
HAIRLINE = "#e1e0d9"
SURFACE = "#fcfcfb"
PAGE = "#f9f9f7"
ACCENT = "#2a78d6"
ACCENT_SOFT = "#cde2fb"
WAIT_GRAY = "#c3c2b7"
GOOD, WARNING, CRITICAL = "#0ca30c", "#fab219", "#d03b3b"
GOOD_TEXT = "#006300"

CSS = f"""
<style>
:root {{ --ink:{INK}; --ink2:{INK_2}; --muted:{MUTED}; --hair:{HAIRLINE}; --surface:{SURFACE}; --accent:{ACCENT}; }}
.stApp {{ background:{PAGE}; }}
.block-container {{ padding-top: 2.2rem; max-width: 1400px; }}
h1, h2, h3 {{ letter-spacing: -0.01em; }}
[data-testid="stSidebar"] {{ background:#f3f2ee; border-right:1px solid var(--hair); }}

.lv-brand {{ display:flex; align-items:baseline; gap:.6rem; margin-bottom:.2rem; }}
.lv-brand .name {{ font-size:1.9rem; font-weight:700; color:var(--ink); }}
.lv-brand .tag {{ color:var(--ink2); font-size:.95rem; }}
.lv-sub {{ color:var(--muted); font-size:.85rem; margin-bottom:1rem; }}

.lv-card {{ background:var(--surface); border:1px solid rgba(11,11,11,.10); border-radius:10px; padding:14px 16px; margin-bottom:12px; }}
.lv-card-title {{ font-size:.72rem; letter-spacing:.06em; text-transform:uppercase; color:var(--muted); margin-bottom:8px; font-weight:600; }}

.lv-pill {{ display:inline-flex; align-items:center; gap:6px; padding:2px 10px; border-radius:999px; font-size:.8rem; font-weight:600;
            border:1px solid rgba(11,11,11,.10); background:#fff; color:var(--ink); }}
.lv-dot {{ width:8px; height:8px; border-radius:50%; display:inline-block; }}

.lv-claims {{ margin:.2rem 0 .4rem 0; padding-left:1.1rem; }}
.lv-claims li {{ margin:.35rem 0; line-height:1.55; color:var(--ink); }}
.lv-chip {{ display:inline-block; font-size:.72rem; font-weight:600; color:{ACCENT}; background:#eef4fc; border:1px solid #cde2fb;
            border-radius:6px; padding:0 6px; margin-left:4px; vertical-align:1px; font-family:ui-monospace,Consolas,monospace; }}
.lv-scope {{ display:inline-block; font-size:.76rem; color:{INK_2}; background:#f1f0ec; border:1px solid {HAIRLINE};
             border-radius:999px; padding:1px 9px; margin-left:6px; }}
.lv-doctype {{ font-size:.74rem; color:{MUTED}; margin:-10px 0 6px 30px; }}
.lv-notice {{ border-left:3px solid {WARNING}; background:#fff8e8; padding:8px 12px; border-radius:6px; color:var(--ink); font-size:.9rem; margin:.4rem 0; }}
.lv-refusal {{ border-left:3px solid {MUTED}; background:#f3f2ee; padding:8px 12px; border-radius:6px; color:var(--ink2); font-size:.92rem; }}

.lv-src {{ border:1px solid rgba(11,11,11,.10); border-radius:8px; padding:10px 12px; background:#fff; margin-bottom:8px; }}
.lv-src-head {{ display:flex; justify-content:space-between; align-items:center; font-size:.82rem; color:var(--ink2); margin-bottom:6px; }}
.lv-src-head b {{ color:var(--ink); }}
.lv-src-body {{ font-size:.88rem; line-height:1.55; color:var(--ink); }}
.lv-src-body mark {{ background:{ACCENT_SOFT}; color:var(--ink); padding:0 2px; border-radius:3px; }}

.lv-kv {{ display:grid; grid-template-columns: 1fr 1fr 1fr; gap:10px; }}
.lv-stat .label {{ font-size:.75rem; color:var(--ink2); }}
.lv-stat .value {{ font-size:1.35rem; font-weight:650; color:var(--ink); line-height:1.2; }}
.lv-stat .hint {{ font-size:.72rem; color:var(--muted); }}

.lv-checks {{ list-style:none; padding:0; margin:0; }}
.lv-checks li {{ display:flex; gap:10px; padding:7px 0; border-top:1px solid var(--hair); font-size:.86rem; }}
.lv-checks li:first-child {{ border-top:none; }}
.lv-checks .icon {{ width:18px; text-align:center; font-weight:700; }}
.lv-checks .what {{ color:var(--ink); font-weight:600; min-width:118px; }}
.lv-checks .detail {{ color:var(--ink2); }}

.lv-track {{ position:relative; height:10px; background:#eceae4; border-radius:5px; margin:22px 0 6px 0; }}
.lv-track .fill {{ position:absolute; left:0; top:0; bottom:0; border-radius:5px; }}
.lv-track .cut {{ position:absolute; top:-6px; bottom:-6px; width:2px; background:var(--ink); }}
.lv-track .cutlabel {{ position:absolute; top:-22px; font-size:.7rem; color:var(--ink2); transform:translateX(-50%); white-space:nowrap; }}
.lv-scale {{ display:flex; justify-content:space-between; font-size:.7rem; color:var(--muted); }}

.lv-mono {{ font-family:ui-monospace,Consolas,monospace; font-size:.78rem; color:var(--ink2); word-break:break-all; }}
.lv-def {{ font-size:.78rem; color:var(--ink2); line-height:1.45; }}
.lv-caveat {{ font-size:.76rem; color:var(--muted); line-height:1.4; margin-top:4px; }}
.lv-pending {{ display:inline-block; font-size:.72rem; font-weight:600; color:{INK_2}; background:#f1f0ec; border:1px dashed #c3c2b7; border-radius:6px; padding:1px 7px; }}

.lv-seg {{ display:flex; height:14px; border-radius:7px; overflow:hidden; gap:2px; background:transparent; margin:8px 0; }}
.lv-legend {{ display:flex; gap:14px; flex-wrap:wrap; font-size:.8rem; color:var(--ink2); }}

.lv-hero-eval {{ padding:6px 0 18px 0; }}
.lv-proof {{ display:grid; grid-template-columns:repeat(3, minmax(0,1fr)); gap:14px; margin-bottom:16px; }}
@media (max-width: 900px) {{ .lv-proof {{ grid-template-columns:repeat(2, minmax(0,1fr)); }} }}
.lv-tile {{ position:relative; background:var(--surface); border:1px solid rgba(11,11,11,.10); border-radius:14px;
            padding:18px 20px; overflow:hidden; }}
.lv-tile::before {{ content:""; position:absolute; left:0; top:0; right:0; height:3px; background:linear-gradient(90deg,{ACCENT},#86b6ef); }}
.lv-tile .v {{ font-size:2.4rem; font-weight:750; color:var(--ink); letter-spacing:-0.03em; line-height:1.05; }}
.lv-tile .k {{ font-size:.95rem; font-weight:650; color:var(--ink); margin-top:6px; }}
.lv-tile .d {{ font-size:.82rem; color:var(--ink2); margin-top:4px; line-height:1.45; }}
.lv-trust {{ display:flex; gap:18px; flex-wrap:wrap; color:var(--ink2); font-size:.85rem; margin:4px 0 22px 0; }}
.lv-trust b {{ color:var(--ink); }}
.lv-hero-title {{ font-size:1.7rem; font-weight:700; color:var(--ink); letter-spacing:-0.02em; }}
.lv-hero-sub {{ color:var(--ink2); font-size:.95rem; margin-top:4px; }}
.lv-hero-sub b {{ color:var(--ink); font-weight:600; }}

.lv-q {{ position:relative; background:var(--surface); border:1px solid rgba(11,11,11,.10); border-radius:14px;
         padding:20px 22px 18px 22px; margin-bottom:16px; overflow:hidden; }}
.lv-q::before {{ content:""; position:absolute; left:0; top:0; right:0; height:3px; background:linear-gradient(90deg,{ACCENT},#86b6ef); }}
.lv-q-num {{ font-size:.75rem; font-weight:700; color:{ACCENT}; letter-spacing:.08em; }}
.lv-q-title {{ font-size:1.12rem; font-weight:650; color:var(--ink); margin:2px 0 14px 0; }}
.lv-bigrow {{ display:flex; gap:28px; flex-wrap:wrap; }}
.lv-big .v {{ font-size:2.1rem; font-weight:700; color:var(--ink); line-height:1.05; letter-spacing:-0.02em; }}
.lv-big .l {{ font-size:.8rem; color:var(--ink2); margin-top:4px; }}
.lv-big .h {{ font-size:.74rem; color:{GOOD_TEXT}; font-weight:600; margin-top:2px; }}
.lv-q-take {{ font-size:.84rem; color:var(--ink2); margin-top:14px; padding-top:12px; border-top:1px solid var(--hair); }}
.lv-status {{ display:flex; gap:10px; flex-wrap:wrap; margin:2px 0 6px 0; }}
.lv-status span {{ font-size:.8rem; color:{GOOD_TEXT}; background:#eef7ee; border:1px solid #cfe8cf; border-radius:999px; padding:3px 11px; font-weight:600; }}

.lv-steps {{ display:flex; gap:10px; align-items:stretch; flex-wrap:wrap; margin:8px 0 18px 0; }}
.lv-step {{ flex:1 1 150px; background:var(--surface); border:1px solid rgba(11,11,11,.10); border-radius:12px; padding:16px; position:relative; }}
.lv-step .n {{ width:26px; height:26px; border-radius:50%; background:{ACCENT}; color:#fff; font-weight:700; font-size:.8rem;
               display:flex; align-items:center; justify-content:center; margin-bottom:10px; }}
.lv-step .t {{ font-weight:650; color:var(--ink); font-size:.95rem; margin-bottom:4px; }}
.lv-step .d {{ color:var(--ink2); font-size:.82rem; line-height:1.45; }}
.lv-step .u {{ color:var(--muted); font-size:.74rem; margin-top:8px; font-family:ui-monospace,Consolas,monospace; }}
.lv-arrow {{ align-self:center; color:#c3c2b7; font-size:1.2rem; }}

[data-testid="stMetric"] {{ background:var(--surface); border:1px solid rgba(11,11,11,.10); border-radius:10px; padding:12px 14px; }}
[data-testid="stMetricLabel"] p {{ font-size:.78rem !important; color:{INK_2}; }}
[data-testid="stMetricValue"] {{ font-size:1.6rem !important; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 4px; border-bottom:1px solid var(--hair); }}
.stTabs [data-baseweb="tab"] {{ padding: 8px 14px; }}
</style>
"""

# status -> (label, colour, icon). Colour never carries meaning alone: icon + label always shown.
STATUS = {
    "answered": ("Answered with sources", GOOD, "✓"),
    "refused_low_confidence": ("Not in your document", MUTED, "∅"),
    "refused_not_in_document": ("Not in your document", MUTED, "∅"),
    "refused_advice": ("Facts only, no advice", WARNING, "!"),
    "off_topic": ("Outside Levi's scope", MUTED, "↷"),
    "error": ("Temporarily unavailable", CRITICAL, "✕"),
}

STAGE_NAMES = {
    "embed_query": "Embed query", "route": "Intent router", "bm25": "BM25 keyword search",
    "dense": "Vector search", "fusion": "RRF fusion", "rerank": "Cross-encoder rerank",
    "expand_context": "Expand context", "overview_build": "Build document overview (one-time)",
    "llm_wait": "Rate-limit wait", "llm": "LLM generation", "llm_repair": "LLM JSON repair",
    "validate": "Validate JSON", "output_check": "Advice check",
}
# Execution order (the timings dict records llm_wait after llm, but the wait happens first).
STAGE_ORDER = ["embed_query", "route", "overview_build", "bm25", "dense", "fusion", "rerank", "expand_context",
               "llm_wait", "llm", "validate", "llm_repair", "output_check"]


def esc(text) -> str:
    return html.escape(str(text))


def pill(status: str, mode: str = "passages") -> str:
    label, colour, icon = STATUS.get(status, (status, MUTED, "•"))
    if status == "answered" and mode == "overview":
        label = "Answered from document overview"
    return f'<span class="lv-pill"><span class="lv-dot" style="background:{colour}"></span>{icon} {esc(label)}</span>'


def highlight(text: str, words: list[str]) -> str:
    safe = esc(text)
    if not words:
        return safe
    pattern = re.compile(r"\b(" + "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True)) + r")\b",
                         flags=re.IGNORECASE)
    return pattern.sub(r"<mark>\1</mark>", safe)


def card(title: str, body_html: str) -> str:
    return f'<div class="lv-card"><div class="lv-card-title">{esc(title)}</div>{body_html}</div>'


def stat(label: str, value: str, hint: str = "") -> str:
    hint_html = f'<div class="hint">{esc(hint)}</div>' if hint else ""
    return f'<div class="lv-stat"><div class="label">{esc(label)}</div><div class="value">{esc(value)}</div>{hint_html}</div>'


def fmt_ms(ms: float) -> str:
    return f"{ms / 1000:.2f} s" if ms >= 1000 else f"{ms:.0f} ms"

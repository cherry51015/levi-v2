"""Visual system for the Levi UI: colour tokens, CSS, and small HTML building blocks.

Palette follows a validated data-viz reference palette: one accent (blue) for
identity, a fixed status palette (good / warning / critical) that always ships
with an icon + text label, and hairline chrome so the content carries the page.
"""
import html
import re

INK = "#101828"
INK_2 = "#475467"
MUTED = "#7a8699"
HAIRLINE = "#dde5f0"
SURFACE = "#ffffff"
PAGE = "#f3f6fb"
ACCENT = "#2a78d6"
ACCENT_SOFT = "#cde2fb"
WAIT_GRAY = "#b8c2d1"
GOOD, WARNING, CRITICAL = "#0ca30c", "#fab219", "#d03b3b"
GOOD_TEXT = "#006300"

CSS = f"""
<style>
:root {{ --ink:{INK}; --ink2:{INK_2}; --muted:{MUTED}; --hair:{HAIRLINE}; --surface:{SURFACE}; --accent:{ACCENT}; }}
.stApp {{ background:{PAGE}; }}
.block-container {{ padding-top: 2.2rem; max-width: 1400px; }}
h1, h2, h3 {{ letter-spacing: -0.01em; }}
[data-testid="stSidebar"] {{ background:#e9eff8; border-right:1px solid var(--hair); }}

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
.lv-scope {{ display:inline-block; font-size:.76rem; color:{INK_2}; background:#eef3fa; border:1px solid {HAIRLINE};
             border-radius:999px; padding:1px 9px; margin-left:6px; }}
.lv-doctype {{ font-size:.74rem; color:{MUTED}; margin:-10px 0 6px 30px; }}
.lv-notice {{ border-left:3px solid {WARNING}; background:#fff8e8; padding:8px 12px; border-radius:6px; color:var(--ink); font-size:.9rem; margin:.4rem 0; }}
.lv-refusal {{ border-left:3px solid {MUTED}; background:#e9eff8; padding:8px 12px; border-radius:6px; color:var(--ink2); font-size:.92rem; }}

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

.lv-track {{ position:relative; height:10px; background:#e3e9f3; border-radius:5px; margin:22px 0 6px 0; }}
.lv-track .fill {{ position:absolute; left:0; top:0; bottom:0; border-radius:5px; }}
.lv-track .cut {{ position:absolute; top:-6px; bottom:-6px; width:2px; background:var(--ink); }}
.lv-track .cutlabel {{ position:absolute; top:-22px; font-size:.7rem; color:var(--ink2); transform:translateX(-50%); white-space:nowrap; }}
.lv-scale {{ display:flex; justify-content:space-between; font-size:.7rem; color:var(--muted); }}

.lv-mono {{ font-family:ui-monospace,Consolas,monospace; font-size:.78rem; color:var(--ink2); word-break:break-all; }}
.lv-def {{ font-size:.78rem; color:var(--ink2); line-height:1.45; }}
.lv-caveat {{ font-size:.76rem; color:var(--muted); line-height:1.4; margin-top:4px; }}
.lv-pending {{ display:inline-block; font-size:.72rem; font-weight:600; color:{INK_2}; background:#eef3fa; border:1px dashed #c3c2b7; border-radius:6px; padding:1px 7px; }}

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

# Visual polish, kept separate from the structural CSS above: typography, depth, the hero banner.
POLISH_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
.stApp, .stApp *:not([data-testid="stIconMaterial"]):not(.material-symbols-rounded) {
  font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1.4rem; }

.lv-hero { position:relative; border-radius:18px; padding:24px 28px; margin:0 0 18px 0; overflow:hidden; color:#fff;
  background: radial-gradient(900px 260px at 92% -30%, rgba(134,182,239,.40), transparent 60%),
              linear-gradient(135deg, #0d1b2e 0%, #13325b 55%, #1c5cab 100%);
  box-shadow: 0 12px 32px rgba(13,27,46,.20); }
.lv-hero .row { display:flex; align-items:center; gap:14px; }
.lv-logo { width:46px; height:46px; border-radius:12px; display:flex; align-items:center; justify-content:center;
  font-size:1.45rem; background:rgba(255,255,255,.12); border:1px solid rgba(255,255,255,.24); }
.lv-hero .name { font-size:1.75rem; font-weight:800; letter-spacing:-.02em; line-height:1.1; }
.lv-hero .tag { color:#cde2fb; font-size:.95rem; font-weight:500; }
.lv-hero .sub { color:#e6eefa; font-size:.92rem; margin-top:12px; max-width:780px; line-height:1.55; }
.lv-hero .badges { display:flex; gap:8px; flex-wrap:wrap; margin-top:14px; }
.lv-hero .badge { font-size:.76rem; font-weight:600; color:#fff; background:rgba(255,255,255,.12);
  border:1px solid rgba(255,255,255,.24); border-radius:999px; padding:4px 11px; }

.lv-card, .lv-q, .lv-tile, .lv-step, .lv-src, [data-testid="stMetric"] {
  box-shadow: 0 1px 2px rgba(16,24,40,.04), 0 6px 20px rgba(16,24,40,.05); }
.lv-card, .lv-src { border-radius:12px; }
.lv-tile, .lv-step { transition: transform .15s ease, box-shadow .15s ease; }
.lv-tile:hover, .lv-step:hover { transform: translateY(-2px);
  box-shadow: 0 2px 4px rgba(16,24,40,.06), 0 14px 30px rgba(16,24,40,.09); }
.lv-tile .v, .lv-hero-title { background: linear-gradient(135deg, #0d1b2e 0%, #1c5cab 100%);
  -webkit-background-clip: text; background-clip: text; color: transparent; }

[data-testid="stChatMessage"] { background:#ffffff; border:1px solid rgba(11,11,11,.08); border-radius:14px;
  padding:14px 16px; margin-bottom:10px; box-shadow: 0 1px 2px rgba(16,24,40,.04), 0 6px 18px rgba(16,24,40,.04); }
[data-testid="stChatInput"] { border-radius:14px; box-shadow: 0 6px 20px rgba(16,24,40,.08); }

.stTabs [data-baseweb="tab-list"] { gap:4px; border-bottom:none; background:#e3e9f3; padding:4px; border-radius:12px;
  width:fit-content; }
.stTabs [data-baseweb="tab"] { border-radius:9px; padding:6px 16px; height:auto; }
.stTabs [aria-selected="true"] { background:#fff; box-shadow: 0 1px 3px rgba(16,24,40,.14); }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display:none; }

[data-testid="stSidebar"] h4 { font-size:.78rem; letter-spacing:.08em; text-transform:uppercase; color:#52514e; }
.lv-sidebrand { display:flex; align-items:center; gap:8px; font-weight:800; font-size:1.05rem; color:#0d1b2e;
  margin:-6px 0 14px 0; }
.lv-sidebrand span { width:28px; height:28px; border-radius:8px; display:flex; align-items:center; justify-content:center;
  background:linear-gradient(135deg,#13325b,#1c5cab); color:#fff; font-size:.9rem; }
/* ---- blue carried through the whole page, not just the banner ---- */
[data-testid="stSidebar"] { background: linear-gradient(180deg, #0d1b2e 0%, #13325b 100%) !important; border-right: none; }
[data-testid="stSidebar"] p, [data-testid="stSidebar"] label, [data-testid="stSidebar"] span,
[data-testid="stSidebar"] small, [data-testid="stSidebar"] div { color: #dbe6f5; }
[data-testid="stSidebar"] h4 { color: #9ec5f4 !important; }
[data-testid="stSidebar"] [data-testid="stCaptionContainer"], [data-testid="stSidebar"] .lv-caveat { color: #9fb3cc !important; }
[data-testid="stSidebar"] .lv-doctype { color: #86b6ef !important; }
[data-testid="stSidebar"] .lv-checks li { border-top-color: rgba(255,255,255,.10); }
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] { background: rgba(255,255,255,.06);
  border: 1px dashed rgba(255,255,255,.28); border-radius: 12px; }
[data-testid="stSidebar"] [data-testid="stFileUploaderFile"] { background: rgba(255,255,255,.06); border-radius: 8px; }
[data-testid="stSidebar"] button { background: rgba(255,255,255,.10); border: 1px solid rgba(255,255,255,.24); color: #fff; }
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.12); }
[data-testid="stSidebar"] [data-testid="stAlert"], [data-testid="stSidebar"] [data-testid="stAlert"] * { color: #101828 !important; }
.lv-sidebrand { color: #ffffff !important; }
.lv-sidebrand span { background: linear-gradient(135deg, #2a78d6, #86b6ef) !important; }

.stTabs [aria-selected="true"] { background: linear-gradient(135deg, #13325b, #1c5cab) !important; }
.stTabs [aria-selected="true"] p { color: #ffffff !important; }

.block-container h4 { border-left: 3px solid #2a78d6; padding-left: 10px; }

[data-testid="stChatMessage"][aria-label="Chat message from user"] { background: #eaf2fd; border-color: #cde2fb; }

.lv-tile.feature { background: linear-gradient(135deg, #0d1b2e 0%, #13325b 55%, #1c5cab 100%); border-color: transparent; }
.lv-tile.feature::before { background: linear-gradient(90deg, #86b6ef, #cde2fb); }
.lv-tile.feature .v { background: none; -webkit-background-clip: initial; background-clip: initial; color: #ffffff; }
.lv-tile.feature .k { color: #ffffff; }
.lv-tile.feature .d { color: #cde2fb; }
.lv-card-title { color: #1c5cab; }
.lv-step .n { background: linear-gradient(135deg, #13325b, #1c5cab); }
</style>
"""

HERO_BADGES = ["Every claim cited", "Says when it doesn't know", "No legal advice", "Open-weight models"]


def hero() -> str:
    badges = "".join(f'<span class="badge">✓ {b}</span>' for b in HERO_BADGES)
    return ('<div class="lv-hero"><div class="row"><div class="lv-logo">⚖️</div><div><div class="name">Levi</div>'
            '<div class="tag">Answers from your contracts, with the receipts</div></div></div>'
            '<div class="sub">Ask anything about your documents. Every answer shows the exact passage it came from, '
            'and Levi tells you when the answer isn\'t there.</div>'
            f'<div class="badges">{badges}</div></div>')


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

"""Design system: colour tokens and the single stylesheet used by every page.

Principles
----------
* One calm neutral base; colour is reserved for meaning (priority, severity,
  detector state, data provenance).
* Colour is never the only signal: every status also has a text label.
* No remote assets.  Fonts are a system stack (Segoe UI on Windows), so the app
  works offline and looks native on the target machine.
"""

from __future__ import annotations

import streamlit as st

# --------------------------------------------------------------------------
# Tokens
# --------------------------------------------------------------------------

INK = "#0B1B2B"  # sidebar / headings on dark
BG = "#F3F5F9"
SURFACE = "#FFFFFF"
BORDER = "#E1E6EE"
BORDER_STRONG = "#C9D1DE"
TEXT = "#0F172A"
MUTED = "#566579"
FAINT = "#8795A8"
BRAND = "#0F766E"
BRAND_SOFT = "#E3F3F0"
ACCENT = "#1D4ED8"

# (foreground, background, border) per semantic tone
TONES: dict[str, tuple[str, str, str]] = {
    "neutral": ("#344054", "#EEF1F6", "#D5DCE8"),
    "brand": ("#0B5F58", "#E3F3F0", "#B5DED8"),
    "info": ("#175CD3", "#E6F0FF", "#BBD5FB"),
    "success": ("#067647", "#DCF5E7", "#AEE3C6"),
    "warning": ("#93370D", "#FEF0C7", "#F6D98B"),
    "danger": ("#B42318", "#FDE7E4", "#F5B8B1"),
    "demo": ("#7A2E0E", "#FFEBD2", "#E8A55B"),
    "muted": ("#566579", "#F2F4F8", "#D5DCE8"),
}

PRIORITY_TONE = {"P1": "danger", "P2": "warning", "P3": "info", "P4": "neutral"}
SEVERITY_TONE = {"critical": "danger", "high": "warning", "medium": "info", "low": "neutral"}
STATE_TONE = {"normal": "success", "watch": "warning", "alert": "danger", "resolved": "neutral"}

# Chart colours (kept distinguishable without relying on hue alone: charts also
# vary shape/dash, and every series is labelled).
SERIES = ["#1D4ED8", "#0F766E", "#B45309", "#7C3AED", "#BE185D", "#475569"]
SEGMENT_COLORS = {"normal": "#94A3B8", "decoy": "#D97706", "planted": "#B42318"}
STATE_COLORS = {"normal": "#2E9E6B", "watch": "#E0A100", "alert": "#C0261B"}
PRIORITY_COLORS = {"P1": "#C0261B", "P2": "#E07B12", "P3": "#3B82F6", "P4": "#94A3B8"}

FONT_STACK = '"Segoe UI", "Segoe UI Variable", system-ui, -apple-system, "Helvetica Neue", Arial, sans-serif'
MONO_STACK = 'Consolas, "Cascadia Mono", "SF Mono", Menlo, "Liberation Mono", monospace'


def _tone_css() -> str:
    rules = []
    for name, (fg, bg, bd) in TONES.items():
        rules.append(f".sr-tone-{name}{{color:{fg};background:{bg};border-color:{bd}}}")
    return "".join(rules)


def build_css() -> str:
    return f"""
<style>
:root {{
  --sr-bg:{BG}; --sr-surface:{SURFACE}; --sr-border:{BORDER}; --sr-border-strong:{BORDER_STRONG};
  --sr-text:{TEXT}; --sr-muted:{MUTED}; --sr-faint:{FAINT}; --sr-brand:{BRAND}; --sr-ink:{INK};
}}
html, body, [class*="css"], .stApp {{ font-family:{FONT_STACK}; }}
.block-container {{ padding-top:1.4rem; padding-bottom:3rem; max-width:1360px; }}
h1, h2, h3, h4 {{ letter-spacing:-0.012em; color:var(--sr-text); }}
[data-testid="stHeader"] {{ background:transparent; }}
[data-testid="stSidebar"] .block-container {{ padding-top:1rem; }}

/* ---- page header ---- */
.sr-header {{ display:flex; align-items:flex-end; justify-content:space-between; gap:1rem; flex-wrap:wrap;
  margin:0 0 1.15rem 0; padding-bottom:.95rem; border-bottom:1px solid var(--sr-border); }}
.sr-title {{ font-size:1.7rem; font-weight:700; color:var(--sr-text); line-height:1.2; margin:0; }}
.sr-subtitle {{ color:var(--sr-muted); font-size:.95rem; margin-top:.3rem; max-width:62rem; }}
.sr-badges {{ display:flex; gap:.4rem; flex-wrap:wrap; align-items:center; }}

/* ---- badges / chips ---- */
.sr-badge {{ display:inline-flex; align-items:center; gap:.35rem; padding:.16rem .6rem; border-radius:999px;
  font-size:.73rem; font-weight:650; letter-spacing:.02em; border:1px solid transparent; white-space:nowrap; line-height:1.35; }}
.sr-badge .dot {{ width:.45rem; height:.45rem; border-radius:50%; background:currentColor; display:inline-block; }}
.sr-chip {{ display:inline-flex; align-items:center; gap:.4rem; padding:.22rem .6rem; border-radius:8px;
  background:#EEF2F8; border:1px solid var(--sr-border); font-size:.84rem; color:var(--sr-text); margin:0 .35rem .35rem 0; }}
.sr-chip b {{ color:var(--sr-muted); font-weight:600; font-size:.72rem; text-transform:uppercase; letter-spacing:.04em; }}
{_tone_css()}

/* ---- cards ---- */
.sr-card {{ background:var(--sr-surface); border:1px solid var(--sr-border); border-radius:12px;
  padding:1rem 1.15rem; box-shadow:0 1px 2px rgba(16,24,40,.04); }}
.sr-card-title {{ font-size:.8rem; font-weight:650; color:var(--sr-muted); text-transform:uppercase; letter-spacing:.05em; margin-bottom:.55rem; }}
.sr-kpi .label {{ font-size:.78rem; font-weight:650; color:var(--sr-muted); text-transform:uppercase; letter-spacing:.05em; }}
.sr-kpi .value {{ font-size:2rem; font-weight:700; color:var(--sr-text); line-height:1.15; margin-top:.25rem; font-variant-numeric:tabular-nums; }}
.sr-kpi .hint {{ font-size:.82rem; color:var(--sr-muted); margin-top:.3rem; }}
.sr-kpi.tone-danger .value {{ color:#B42318; }}
.sr-kpi.tone-warning .value {{ color:#93370D; }}
.sr-kpi.tone-success .value {{ color:#067647; }}

.sr-section {{ margin:1.5rem 0 .6rem 0; }}
.sr-section h3 {{ font-size:1.05rem; font-weight:700; margin:0; }}
.sr-section p {{ color:var(--sr-muted); font-size:.88rem; margin:.15rem 0 0 0; }}

/* ---- provenance banners ---- */
.sr-demo-banner {{ background:repeating-linear-gradient(135deg,#FFF3E0,#FFF3E0 14px,#FFE8CC 14px,#FFE8CC 28px);
  border:1.5px dashed #D97706; color:#7A2E0E; border-radius:10px; padding:.7rem 1rem; font-size:.88rem; margin:0 0 1.1rem 0; }}
.sr-demo-banner b {{ letter-spacing:.04em; }}
.sr-callout {{ border-radius:10px; padding:.7rem 1rem; font-size:.88rem; border:1px solid; margin:.2rem 0 1rem 0; }}

/* ---- empty states ---- */
.sr-empty {{ text-align:center; padding:2rem 1.5rem; border:1.5px dashed var(--sr-border-strong); border-radius:14px; background:var(--sr-surface); }}
.sr-empty .icon {{ font-size:1.7rem; line-height:1; }}
.sr-empty h4 {{ margin:.55rem 0 .25rem 0; font-size:1.05rem; }}
.sr-empty p {{ color:var(--sr-muted); font-size:.9rem; max-width:40rem; margin:.2rem auto; }}

/* ---- pipeline stages ---- */
.sr-flow {{ display:flex; gap:.5rem; flex-wrap:wrap; align-items:stretch; }}
.sr-flow .step {{ flex:1 1 150px; background:var(--sr-surface); border:1px solid var(--sr-border); border-radius:12px; padding:.75rem .9rem; position:relative; }}
.sr-flow .step .name {{ font-weight:700; font-size:.85rem; letter-spacing:.06em; color:var(--sr-ink); }}
.sr-flow .step .meta {{ font-size:.8rem; color:var(--sr-muted); margin-top:.25rem; }}
.sr-flow .step .bar {{ height:5px; border-radius:99px; background:#E5EAF1; margin-top:.55rem; overflow:hidden; }}
.sr-flow .step .bar > span {{ display:block; height:100%; background:{BRAND}; }}
.sr-stage {{ display:flex; gap:.9rem; background:var(--sr-surface); border:1px solid var(--sr-border); border-radius:12px; padding:.9rem 1.05rem; margin-bottom:.7rem; }}
.sr-stage .num {{ flex:0 0 2rem; height:2rem; border-radius:50%; background:var(--sr-ink); color:#fff; display:flex; align-items:center; justify-content:center; font-weight:700; font-size:.88rem; }}
.sr-stage.off .num {{ background:#C3CBD8; }}
.sr-stage .body {{ flex:1 1 auto; min-width:0; }}
.sr-stage .head {{ display:flex; justify-content:space-between; gap:.6rem; flex-wrap:wrap; align-items:center; margin-bottom:.35rem; }}
.sr-stage .head .t {{ font-weight:700; font-size:.98rem; }}
.sr-stage .note {{ color:var(--sr-muted); font-size:.87rem; }}

/* ---- text blocks, bars ---- */
.sr-textblock {{ background:#F8FAFC; border:1px solid var(--sr-border); border-radius:10px; padding:.75rem .9rem;
  font-size:.92rem; line-height:1.5; white-space:pre-wrap; word-break:break-word; color:var(--sr-text); min-height:3rem; }}
.sr-textblock.mono {{ font-family:{MONO_STACK}; font-size:.88rem; }}
.sr-label {{ font-size:.74rem; font-weight:650; color:var(--sr-muted); text-transform:uppercase; letter-spacing:.05em; margin:0 0 .3rem 0; }}
.sr-rule {{ display:flex; gap:.5rem; align-items:center; flex-wrap:wrap; padding:.35rem 0; border-bottom:1px solid #EEF1F6; font-size:.88rem; }}
.sr-rule:last-child {{ border-bottom:0; }}
.sr-rule code {{ font-family:{MONO_STACK}; background:#EEF2F8; padding:.08rem .4rem; border-radius:5px; font-size:.82rem; }}
.sr-rule .arrow {{ color:var(--sr-faint); }}
.sr-rule .rid {{ margin-left:auto; color:var(--sr-faint); font-size:.74rem; font-family:{MONO_STACK}; }}
.sr-conf {{ margin-top:.2rem; }}
.sr-conf .track {{ height:8px; background:#E5EAF1; border-radius:99px; position:relative; }}
.sr-conf .fill {{ height:100%; border-radius:99px; }}
.sr-conf .mark {{ position:absolute; top:-4px; width:2px; height:16px; background:var(--sr-ink); }}
.sr-conf .cap {{ display:flex; justify-content:space-between; font-size:.78rem; color:var(--sr-muted); margin-top:.3rem; }}
.sr-kv {{ display:flex; justify-content:space-between; gap:1rem; padding:.38rem 0; border-bottom:1px solid #EEF1F6; font-size:.88rem; }}
.sr-kv:last-child {{ border-bottom:0; }}
.sr-kv .k {{ color:var(--sr-muted); }}
.sr-kv .v {{ font-weight:600; text-align:right; }}
.sr-mini {{ font-size:.82rem; color:var(--sr-muted); }}

/* ---- sidebar brand ---- */
.sr-brand {{ display:flex; align-items:center; gap:.7rem; padding:.2rem 0 .9rem 0; }}
.sr-radar {{ width:38px; height:38px; border-radius:50%; flex:0 0 38px; border:2px solid #2DD4BF;
  background:conic-gradient(from 200deg, rgba(45,212,191,0) 0deg, rgba(45,212,191,.55) 70deg, rgba(45,212,191,0) 72deg),
  radial-gradient(circle, #2DD4BF 0 2.5px, transparent 3px),
  radial-gradient(circle, transparent 0 8px, rgba(45,212,191,.55) 8px 9px, transparent 9px 14px, rgba(45,212,191,.35) 14px 15px, transparent 15px); }}
.sr-brand .name {{ font-weight:750; font-size:1.05rem; letter-spacing:.01em; }}
.sr-brand .tag {{ font-size:.72rem; opacity:.75; margin-top:.05rem; }}
.sr-side-note {{ font-size:.78rem; opacity:.8; line-height:1.4; }}

/* tables */
[data-testid="stDataFrame"] {{ border:1px solid var(--sr-border); border-radius:10px; overflow:hidden; }}
</style>
"""


def inject_css() -> None:
    """Inject the stylesheet.  Cheap enough to call on every rerun."""
    st.markdown(build_css(), unsafe_allow_html=True)

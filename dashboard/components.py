"""Reusable presentation components (HTML snippets + small render helpers).

Safety: every dynamic value is escaped through ``esc``.  Ticket text will come
from CSV files, so it must never be interpreted as HTML or Markdown.

Markdown note: the HTML is built on a single line (no blank lines, no leading
indentation) because Markdown would otherwise end the HTML block or turn it into
a code block.
"""

from __future__ import annotations

import html
import re
from collections.abc import Iterable, Sequence

import streamlit as st

from dashboard.contracts import PRIORITY_LABELS
from dashboard.theme import PRIORITY_TONE, SEVERITY_TONE, STATE_TONE

_GAP_RE = re.compile(r">\s+<")


def esc(value: object) -> str:
    """Escape for safe inclusion in HTML *and* inert under Markdown/LaTeX."""
    text = html.escape(str(value), quote=True)
    return text.replace("\n", "&#10;").replace("$", "&#36;").replace("*", "&#42;").replace("_", "&#95;")


def _compact(markup: str) -> str:
    return _GAP_RE.sub("><", markup.strip().replace("\n", " "))


def render(markup: str) -> None:
    st.markdown(_compact(markup), unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Badges
# --------------------------------------------------------------------------


def badge(label: str, tone: str = "neutral", dot: bool = False) -> str:
    marker = '<span class="dot"></span>' if dot else ""
    return f'<span class="sr-badge sr-tone-{tone}">{marker}{esc(label)}</span>'


def priority_badge(priority: str) -> str:
    label = PRIORITY_LABELS.get(priority)
    text = f"{priority} {label}" if label else str(priority)
    return badge(text, PRIORITY_TONE.get(priority, "neutral"), dot=True)


def severity_badge(severity: str) -> str:
    return badge(str(severity).capitalize(), SEVERITY_TONE.get(str(severity), "neutral"), dot=True)


def state_badge(state: str) -> str:
    return badge(str(state).upper(), STATE_TONE.get(str(state), "neutral"), dot=True)


def provenance_badge(kind: str) -> str:
    """kind: 'live' | 'demo' | 'off'"""
    if kind == "live":
        return badge("LIVE", "success", dot=True)
    if kind == "demo":
        return badge("DEMO DATA", "demo", dot=True)
    return badge("NOT CONNECTED", "muted", dot=True)


def review_badge(needs_review: bool) -> str:
    return badge("Human review", "warning", dot=True) if needs_review else badge("Auto-routed", "success", dot=True)


# --------------------------------------------------------------------------
# Layout blocks
# --------------------------------------------------------------------------


def page_header(title: str, subtitle: str | None = None, badges: Iterable[str] = ()) -> None:
    sub = f'<div class="sr-subtitle">{esc(subtitle)}</div>' if subtitle else ""
    badge_html = "".join(badges)
    render(
        f'<div class="sr-header"><div><div class="sr-title">{esc(title)}</div>{sub}</div>'
        f'<div class="sr-badges">{badge_html}</div></div>'
    )


def section(title: str, caption: str | None = None) -> None:
    cap = f"<p>{esc(caption)}</p>" if caption else ""
    render(f'<div class="sr-section"><h3>{esc(title)}</h3>{cap}</div>')


def demo_banner(detail: str = "") -> None:
    extra = f" {esc(detail)}" if detail else ""
    render(
        '<div class="sr-demo-banner"><b>DEMO DATA</b> &mdash; illustrative synthetic data used only to '
        "preview this design. These are <b>not</b> model predictions, detector output or measured "
        f"results.{extra}</div>"
    )


def callout(text: str, tone: str = "info") -> None:
    render(f'<div class="sr-callout sr-tone-{tone}">{esc(text)}</div>')


def kpi_card(label: str, value: object, hint: str | None = None, tone: str | None = None) -> str:
    tone_cls = f" tone-{tone}" if tone else ""
    hint_html = f'<div class="hint">{esc(hint)}</div>' if hint else ""
    return (
        f'<div class="sr-card sr-kpi{tone_cls}"><div class="label">{esc(label)}</div>'
        f'<div class="value">{esc(value)}</div>{hint_html}</div>'
    )


def kpi_row(items: Sequence[tuple]) -> None:
    """items: (label, value[, hint[, tone]])"""
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        with col:
            render(kpi_card(*item))


def card(title: str | None, body_html: str) -> str:
    head = f'<div class="sr-card-title">{esc(title)}</div>' if title else ""
    return f'<div class="sr-card">{head}{body_html}</div>'


def empty_state(title: str, body: str, interface: str | None = None, icon: str = "◌") -> None:
    render(
        f'<div class="sr-empty"><div class="icon">{esc(icon)}</div><h4>{esc(title)}</h4>'
        f"<p>{esc(body)}</p></div>"
    )
    if interface:
        with st.expander("Backend interface this page is waiting for"):
            st.code(interface, language="python")


def chips(pairs: Iterable[tuple[str, str]]) -> str:
    return "".join(f'<span class="sr-chip"><b>{esc(k)}</b>{esc(v)}</span>' for k, v in pairs)


def kv_list(rows: Iterable[tuple[str, str]], raw_values: bool = False) -> str:
    out = []
    for key, value in rows:
        shown = value if raw_values else esc(value)
        out.append(f'<div class="sr-kv"><span class="k">{esc(key)}</span><span class="v">{shown}</span></div>')
    return "".join(out)


def text_block(label: str, text: str, mono: bool = False) -> str:
    cls = "sr-textblock mono" if mono else "sr-textblock"
    shown = esc(text) if text else '<span style="color:#8795A8">(empty)</span>'
    return f'<div class="sr-label">{esc(label)}</div><div class="{cls}">{shown}</div>'


def confidence_bar(value: float, threshold: float | None = None) -> str:
    pct = max(0.0, min(1.0, float(value))) * 100
    colour = "#2E9E6B" if threshold is None or value >= threshold else "#E0A100"
    mark = f'<div class="mark" style="left:{threshold * 100:.1f}%"></div>' if threshold is not None else ""
    cap = f"<span>Confidence {pct:.0f}%</span>"
    if threshold is not None:
        cap += f"<span>Review threshold {threshold * 100:.0f}%</span>"
    return (
        f'<div class="sr-conf"><div class="track"><div class="fill" style="width:{pct:.1f}%;background:{colour}"></div>'
        f'{mark}</div><div class="cap">{cap}</div></div>'
    )


def stage_card(number: int, title: str, provenance: str, body_html: str) -> str:
    off = " off" if provenance == "off" else ""
    return (
        f'<div class="sr-stage{off}"><div class="num">{number}</div><div class="body">'
        f'<div class="head"><span class="t">{esc(title)}</span>{provenance_badge(provenance)}</div>'
        f"{body_html}</div></div>"
    )


def workflow_strip(stage_status: Sequence[tuple[str, int, int]]) -> None:
    """stage_status: (stage name, connected modules, total modules)"""
    steps = []
    for name, connected, total in stage_status:
        pct = 0 if not total else round(100 * connected / total)
        meta = "no modules" if not total else f"{connected} of {total} connected"
        steps.append(
            f'<div class="step"><div class="name">{esc(name)}</div><div class="meta">{esc(meta)}</div>'
            f'<div class="bar"><span style="width:{pct}%"></span></div></div>'
        )
    render(f'<div class="sr-flow">{"".join(steps)}</div>')


def mini(text: str) -> None:
    render(f'<div class="sr-mini">{esc(text)}</div>')

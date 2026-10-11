"""App shell: page config, styling, sidebar and navigation."""

from __future__ import annotations

import streamlit as st

from dashboard import backend
from dashboard.data_access import PREVIEW_KEY
from dashboard.theme import inject_css

APP_NAME = "SupportAI Radar"


def bootstrap(preview: bool | None = None) -> None:
    """Styling + session defaults.  ``preview`` is for tests and scripts."""
    inject_css()
    st.session_state.setdefault(PREVIEW_KEY, False)
    if preview is not None:
        st.session_state[PREVIEW_KEY] = preview


def _sidebar() -> None:
    with st.sidebar:
        st.markdown(
            '<div class="sr-brand"><div class="sr-radar"></div><div><div class="name">SupportAI Radar</div>'
            '<div class="tag">Support routing &amp; incident detection</div></div></div>',
            unsafe_allow_html=True,
        )
    # navigation renders here (st.navigation places it in the sidebar)


def _sidebar_footer() -> None:
    with st.sidebar:
        st.divider()
        st.toggle(
            "Design preview (DEMO DATA)",
            key=PREVIEW_KEY,
            help="Shows the intended design with clearly labelled synthetic data on pages whose backend is not connected yet. "
                 "Live backend data always takes precedence.",
        )
        caps = backend.capabilities()
        connected = sum(c.connected for c in caps)
        with st.expander(f"Backend status · {connected} of {len(caps)} connected"):
            for c in caps:
                st.markdown(f"{'●' if c.connected else '○'} **{c.label}**  \n<span class='sr-side-note'>{c.detail}</span>", unsafe_allow_html=True)
        st.markdown(
            '<div class="sr-side-note">Local prototype. Runs on this machine only; no external services.<br>'
            "Alerts are potential incidents inferred from ticket volume, not confirmed outages.</div>",
            unsafe_allow_html=True,
        )


def run() -> None:
    from dashboard.pages import analysis, evaluation, explorer, incidents, overview, replay

    st.set_page_config(page_title=APP_NAME, page_icon="📡", layout="wide", initial_sidebar_state="expanded")
    bootstrap()
    _sidebar()
    pages = [
        st.Page(overview.render, title="Operations Overview", icon=":material/dashboard:", url_path="overview", default=True),
        st.Page(explorer.render, title="Ticket Explorer", icon=":material/manage_search:", url_path="tickets"),
        st.Page(analysis.render, title="Ticket Analysis", icon=":material/psychology:", url_path="analysis"),
        st.Page(incidents.render, title="Incident Center", icon=":material/crisis_alert:", url_path="incidents"),
        st.Page(replay.render, title="Incident Replay", icon=":material/replay:", url_path="replay"),
        st.Page(evaluation.render, title="Evaluation", icon=":material/fact_check:", url_path="evaluation"),
    ]
    navigation = st.navigation(pages, position="sidebar")
    _sidebar_footer()
    navigation.run()

"""Small Streamlit version shims, so the UI works across recent releases."""

from __future__ import annotations

import inspect

import streamlit as st


def _supports_width_argument() -> bool:
    try:
        return "width" in inspect.signature(st.dataframe).parameters
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return False


_STRETCH: dict = {"width": "stretch"} if _supports_width_argument() else {"use_container_width": True}


def stretch() -> dict:
    """Keyword arguments that make a widget fill its container.

    Newer Streamlit uses ``width="stretch"``; older uses ``use_container_width=True``.
    """
    return dict(_STRETCH)

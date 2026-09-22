"""Per-session prediction store (replaces the React client store).

The Result and Explainability (local) pages read the most recent assessment the
user submitted on the Assessment page. Streamlit's session_state is per-browser
session, matching the old device-local behaviour.
"""
from __future__ import annotations

from typing import Any, Optional

import streamlit as st

_KEY = "last_prediction"


def save_prediction(payload: dict[str, Any], result: dict[str, Any]) -> None:
    st.session_state[_KEY] = {"payload": payload, "result": result}


def get_prediction() -> Optional[dict[str, Any]]:
    """Returns {'payload': ..., 'result': ...} or None if nothing assessed yet."""
    return st.session_state.get(_KEY)


def clear_prediction() -> None:
    st.session_state.pop(_KEY, None)

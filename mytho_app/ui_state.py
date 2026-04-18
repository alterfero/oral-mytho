from __future__ import annotations

from collections.abc import MutableMapping


def pending_widget_reset_key(widget_key: str) -> str:
    return f"{widget_key}__reset_pending"


def mark_widget_for_reset(session_state: MutableMapping[str, object], widget_key: str) -> None:
    session_state[pending_widget_reset_key(widget_key)] = True


def apply_pending_widget_reset(
    session_state: MutableMapping[str, object],
    widget_key: str,
    *,
    reset_value: str = "",
) -> None:
    pending_key = pending_widget_reset_key(widget_key)
    if session_state.pop(pending_key, False):
        session_state[widget_key] = reset_value


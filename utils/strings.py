from __future__ import annotations


def sanitize(value: str) -> str:
    return "".join(c if (c.isalnum() or c in "-._") else "-" for c in value)

from __future__ import annotations

import sys


def render_progress(
    label: str,
    current: int,
    total: int,
    *,
    detail: str | None = None,
    width: int = 28,
) -> None:
    total = max(0, int(total))
    current = max(0, int(current))
    if total == 0:
        ratio = 1.0
        current = 0
    else:
        current = min(current, total)
        ratio = current / total

    filled = min(width, int(round(width * ratio)))
    bar = "#" * filled + "-" * (width - filled)
    detail_text = f" {detail}" if detail else ""
    line = (
        f"\r{label:<16} [{bar}] {current}/{total} "
        f"{ratio * 100:6.2f}%{detail_text}"
    )
    sys.stdout.write(line + "        ")
    if total == 0 or current >= total:
        sys.stdout.write("\n")
    sys.stdout.flush()

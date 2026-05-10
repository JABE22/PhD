"""Sync appendix figure assets into papers/media/appendices.

Run from project root:
    python3 papers/tables/sync_appendix_media.py
"""

from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
APPENDIX_MEDIA = ROOT / "papers" / "media" / "appendices"
APPENDIX_MEDIA.mkdir(parents=True, exist_ok=True)

# (source relative to project root, destination filename in papers/media/appendices)
FIGURE_MAP: list[tuple[str, str]] = [
    ("results/test1/extra_visualizations_row1.png", "t1_extra_visualizations_row1.png"),
    ("results/test1/extra_visualizations_row2.png", "t1_extra_visualizations_row2.png"),
    ("results/test1/extra_visualizations_dashboard.png", "t1_extra_visualizations_dashboard.png"),

]


def main() -> None:
    copied: list[str] = []
    missing: list[str] = []

    for src_rel, dst_name in FIGURE_MAP:
        src = ROOT / src_rel
        dst = APPENDIX_MEDIA / dst_name
        if src.exists():
            shutil.copy2(src, dst)
            copied.append(dst_name)
        else:
            missing.append(src_rel)

    if copied:
        print("Copied appendix figures:")
        for name in copied:
            print(f" - papers/media/appendices/{name}")

    if missing:
        print("Missing source files (not copied):")
        for rel in missing:
            print(f" - {rel}")


if __name__ == "__main__":
    main()

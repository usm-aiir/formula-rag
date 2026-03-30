from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional

MATHIMAGES_ROOT = Path("/home/lucas.matheson/MathImages")

# Each source: (name, images_dir, tsv_path)
SOURCES = [
    ("MSE",          MATHIMAGES_ROOT / "MSEImages",          MATHIMAGES_ROOT / "MSEImages"          / "MSE.tsv"),
    ("MathOverflow", MATHIMAGES_ROOT / "MathOverFlowImages", MATHIMAGES_ROOT / "MathOverFlowImages" / "math_overflow.tsv"),
    ("Mathematica",  MATHIMAGES_ROOT / "MathematicaImages",  MATHIMAGES_ROOT / "MathematicaImages"  / "mathematica.tsv"),
]


@dataclass
class MathImageEntry:
    source: str          # "MSE" | "MathOverflow" | "Mathematica"
    image_id: str        #  "97_2"  (post_id + _ + image_index)
    post_id: str         # "97"
    image_index: int     #  2
    title: str           # question title (may contain LaTeX)
    url: str             # link to the source post
    image_path: Path     # absolute path to the .png file


def _parse_tsv(source: str, images_dir: Path, tsv_path: Path) -> Iterator[MathImageEntry]:
    """Yield MathImageEntry for every row in a source TSV."""
    with open(tsv_path, encoding="utf-8", newline="") as f:
        for row in csv.reader(f, delimiter="\t"):
            if len(row) < 4:
                continue
            image_id, title, _, url = row[0], row[1], row[2], row[3]
            image_path = images_dir / f"{image_id}.png"

            parts = image_id.rsplit("_", 1)
            post_id = parts[0]
            image_index = int(parts[1]) if len(parts) == 2 and parts[1].isdigit() else 0

            yield MathImageEntry(
                source=source,
                image_id=image_id,
                post_id=post_id,
                image_index=image_index,
                title=title,
                url=url,
                image_path=image_path,
            )


def iter_dataset(
    sources: Optional[List[str]] = None,
    only_existing: bool = True,
) -> Iterator[MathImageEntry]:
    """
    Iterate over all entries in the MathImages dataset.

    Args:
        sources:       Subset of source names to include. None = all three.
        only_existing: Skip entries whose image file does not exist on disk.
    """
    active = {name for name, _, _ in SOURCES} if sources is None else set(sources)

    for name, images_dir, tsv_path in SOURCES:
        if name not in active:
            continue
        for entry in _parse_tsv(name, images_dir, tsv_path):
            if only_existing and not entry.image_path.exists():
                continue
            yield entry


if __name__ == "__main__":
    total = 0
    source_counts: dict[str, int] = {}

    for entry in iter_dataset():
        source_counts[entry.source] = source_counts.get(entry.source, 0) + 1
        total += 1

        if total <= 5:
            print(f"[{entry.source}] {entry.image_id}  post={entry.post_id}  idx={entry.image_index}")
            print(f"  title : {entry.title[:80]}")
            print(f"  url   : {entry.url}")
            print(f"  image : {entry.image_path}")
            print()

    print(f"Total entries: {total}")
    for src, count in source_counts.items():
        print(f"  {src}: {count}")

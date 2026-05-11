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
    image_id: str        #  Example: "97_2"  its just the post_id + _ + image_index
    post_id: str       
    image_index: int   
    title: str          
    url: str  # link to the source post
    image_path: Path # absolute path to the .png file


def _parse_tsv(source: str, images_dir: Path, tsv_path: Path) -> Iterator[MathImageEntry]:
    with open(tsv_path, encoding="utf-8", newline="") as f:
        for row in csv.reader(f, delimiter="\t"):
            if len(row) < 4:
                continue
            image_id, title, _, url = row[0], row[1], row[2], row[3]
            image_path = images_dir / f"{image_id}.png"

            # image_id encodes "<post_id>_<image_index>" (e.g. "97_2")
            post_id, image_index = image_id.rsplit("_", 1)
            yield MathImageEntry(
                source=source,
                image_id=image_id,
                post_id=post_id,
                image_index=int(image_index),
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
    # default to all sources
    if sources is None:
        active = {name for name, _, _ in SOURCES}
    else:
        active = set(sources)

    for name, images_dir, tsv_path in SOURCES:
        if name not in active:
            continue
        yield from (
            entry
            for entry in _parse_tsv(name, images_dir, tsv_path)
            if not only_existing or entry.image_path.exists()
        )


if __name__ == "__main__":
    source_counts: dict[str, int] = {}

    for index, entry in enumerate(iter_dataset()):
        source_counts[entry.source] = source_counts.get(entry.source, 0) + 1

        if index < 5:
            print(f"[{entry.source}] {entry.image_id}  post={entry.post_id}  idx={entry.image_index}")
            print(f"  title : {entry.title[:80]}")
            print(f"  url   : {entry.url}")
            print(f"  image : {entry.image_path}")
            print()

    print(f"Total entries: {index + 1}")
    for src, count in source_counts.items():
        print(f"  {src}: {count}")

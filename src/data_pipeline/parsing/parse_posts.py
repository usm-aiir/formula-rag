"""
Parse Posts.V1.3.xml from the ARQMath collection.

Streams the 4.1 GB XML file using iterparse so it never loads into memory.
For each post, extracts:
  - plain text (HTML stripped, math-container spans removed)
  - LaTeX formulas (from <span class="math-container"> tags)
  - image URLs (from <img src="..."> tags in the body)

Output schema (one JSON object per line):
  {
    "post_id": int,
    "post_type": "question" | "answer",
    "parent_id": int | null,          # answer → parent question id
    "accepted_answer_id": int | null,  # question → accepted answer id
    "title": str | null,               # questions only
    "tags": list[str],                 # questions only
    "score": int,
    "creation_date": str,
    "text": str,                       # clean text, no LaTeX, no HTML
    "formulas": list[str],             # LaTeX strings in order of appearance
    "image_urls": list[str]            # raw image src URLs
  }
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Iterator
from xml.etree import ElementTree as ET

from bs4 import BeautifulSoup
from tqdm import tqdm


# ARQMath marks all formulas with this class
MATH_CLASS = "math-container"


def _parse_body(html: str) -> tuple[str, list[str], list[str]]:
    """
    Parse an MSE post body (HTML string).

    Returns:
        text       — clean text with formulas and HTML removed
        formulas   — list of LaTeX strings in order of appearance
        image_urls — list of image src URLs
    """
    soup = BeautifulSoup(html, "lxml")

    formulas: list[str] = []
    image_urls: list[str] = []

    # Collect images before we start removing things
    for img in soup.find_all("img"):
        src = img.get("src", "").strip()
        if src:
            image_urls.append(src)

    # Extract LaTeX from math-container spans, then replace each with a
    # placeholder so the surrounding text stays readable.
    for span in soup.find_all("span", class_=MATH_CLASS):
        latex = span.get_text()
        # Strip surrounding $ or $$ delimiters if present
        latex = re.sub(r"^\$+|\$+$", "", latex).strip()
        if latex:
            formulas.append(latex)
        span.replace_with(" ")

    # Remove all remaining HTML tags, collapse whitespace
    text = soup.get_text(separator=" ")
    text = re.sub(r"\s+", " ", text).strip()

    return text, formulas, image_urls


def _parse_tags(raw: str) -> list[str]:
    """'<calculus><integration>' → ['calculus', 'integration']"""
    return re.findall(r"<([^>]+)>", raw)


def iter_posts(xml_path: Path) -> Iterator[dict]:
    """
    Stream Posts.V1.3.xml and yield one dict per post.
    Uses iterparse to avoid loading the full 4.1 GB file into memory.
    """
    context = ET.iterparse(str(xml_path), events=("end",))

    for _event, elem in context:
        if elem.tag != "row":
            continue

        attrib = elem.attrib
        post_type_id = int(attrib.get("PostTypeId", 0))

        if post_type_id not in (1, 2):
            elem.clear()
            continue

        body_html = attrib.get("Body", "")
        text, formulas, image_urls = _parse_body(body_html)

        post: dict = {
            "post_id": int(attrib["Id"]),
            "post_type": "question" if post_type_id == 1 else "answer",
            "parent_id": int(attrib["ParentId"]) if "ParentId" in attrib else None,
            "accepted_answer_id": (
                int(attrib["AcceptedAnswerId"])
                if "AcceptedAnswerId" in attrib
                else None
            ),
            "title": attrib.get("Title", None),
            "tags": _parse_tags(attrib.get("Tags", "")),
            "score": int(attrib.get("Score", 0)),
            "creation_date": attrib.get("CreationDate", ""),
            "text": text,
            "formulas": formulas,
            "image_urls": image_urls,
        }

        yield post

        # Free memory — critical for large files
        elem.clear()


def parse_posts(
    xml_path: Path,
    out_path: Path,
    *,
    images_only: bool = False,
) -> dict[str, int]:
    """
    Parse the full posts XML and write one JSON object per line to out_path.

    Args:
        xml_path    — path to Posts.V1.3.xml
        out_path    — output .jsonl path
        images_only — if True, only write posts that contain at least one image

    Returns:
        counts dict with total, questions, answers, with_images, written
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    counts = {"total": 0, "questions": 0, "answers": 0, "with_images": 0, "written": 0}

    with out_path.open("w", encoding="utf-8") as fout:
        for post in tqdm(iter_posts(xml_path), desc="Parsing posts", unit=" posts"):
            counts["total"] += 1
            if post["post_type"] == "question":
                counts["questions"] += 1
            else:
                counts["answers"] += 1

            if post["image_urls"]:
                counts["with_images"] += 1

            if images_only and not post["image_urls"]:
                continue

            fout.write(json.dumps(post, ensure_ascii=False) + "\n")
            counts["written"] += 1

    return counts


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parse ARQMath Posts XML to JSONL")
    parser.add_argument("xml_path", type=Path, help="Path to Posts.V1.3.xml")
    parser.add_argument("out_path", type=Path, help="Output .jsonl file")
    parser.add_argument(
        "--images-only",
        action="store_true",
        help="Only include posts that contain at least one image",
    )
    args = parser.parse_args()

    counts = parse_posts(args.xml_path, args.out_path, images_only=args.images_only)

    print("\nDone.")
    for k, v in counts.items():
        print(f"  {k}: {v:,}")

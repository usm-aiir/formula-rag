"""
ARQMath Task 1 (Answer Retrieval) data loading.

Loads topics (Title + Question), qrels (topic → post_id → grade), and
post corpus.  Reads directly from the raw ARQMath Posts XML to preserve
inline formula positions — the processed posts.jsonl has formulas stripped
from the body and appended as a separate list, which destroys text coherence.
Falls back to posts.jsonl when the XML is unavailable.
"""

from __future__ import annotations

import html
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_POSTS_JSONL = _PROJECT_ROOT / "data/processed/posts.jsonl"
_POSTS_XML = _PROJECT_ROOT / "data/raw/arqmath/collection/Posts.V1.3.xml"

_QREL_PATHS: Dict[str, List[Path]] = {
    "train": [
        _PROJECT_ROOT / "data/raw/arqmath/qrels/task1/arqmath1/qrel_task1_2020_official",
        _PROJECT_ROOT / "data/raw/arqmath/qrels/task1/arqmath2/qrel_task1_2021_official.tsv",
    ],
    "eval": [
        _PROJECT_ROOT / "data/raw/arqmath/qrels/task1/arqmath3/qrel_task1_2022_official.tsv",
    ],
}

_TOPIC_PATHS: Dict[str, List[Path]] = {
    "train": [
        _PROJECT_ROOT / "data/raw/arqmath/topics/task1/arqmath1/Topics_V2.0.xml",
        _PROJECT_ROOT / "data/raw/arqmath/topics/task1/arqmath2/Topics_Task1_2021_V1.1.xml",
    ],
    "eval": [
        _PROJECT_ROOT / "data/raw/arqmath/topics/task1/arqmath3/Topics_Task1_2022_V0.1.xml",
    ],
}

POSITIVE_GRADE = 1.0

# ---------------------------------------------------------------------------
# HTML / LaTeX conversion
# ---------------------------------------------------------------------------

_MATH_SPAN_RE = re.compile(
    r'<span\s+class="math-container"[^>]*>(.*?)</span>',
    re.DOTALL,
)
_DISPLAY_MATH_RE = re.compile(r"\$\$([^$]+)\$\$")
_INLINE_MATH_RE = re.compile(r"(?<!\$)\$([^$\n]+)\$(?!\$)")


def _math_span_to_token(m: re.Match) -> str:
    """Replace a <span class="math-container"> with [MATH]...[/MATH]."""
    content = m.group(1).strip()
    if content.startswith("$$") and content.endswith("$$"):
        content = content[2:-2].strip()
    elif content.startswith("$") and content.endswith("$"):
        content = content[1:-1].strip()
    return f"[MATH]{content}[/MATH]"


def _html_to_text_with_math(html_str: str) -> str:
    """Convert HTML to plain text, preserving math spans as [MATH]...[/MATH]."""
    if not html_str:
        return ""
    text = _MATH_SPAN_RE.sub(_math_span_to_token, html_str)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def latex_to_math_format(text: str) -> str:
    """Convert bare $...$ and $$...$$ to [MATH]...[/MATH] (fallback path)."""
    if not text:
        return ""
    text = _DISPLAY_MATH_RE.sub(r"[MATH]\1[/MATH]", text)
    text = _INLINE_MATH_RE.sub(r"[MATH]\1[/MATH]", text)
    return text


def _strip_html(html_str: str) -> str:
    """Remove HTML tags and decode entities (no math awareness)."""
    if not html_str:
        return ""
    text = re.sub(r"<[^>]+>", " ", html_str)
    return html.unescape(text).strip()


# ---------------------------------------------------------------------------
# Topics & qrels
# ---------------------------------------------------------------------------

def load_topics(split: str) -> Dict[str, str]:
    """
    Load Task 1 topics.  Returns {topic_id: query_text}.
    Uses math-aware HTML stripping so inline formulas become [MATH]...[/MATH].
    """
    topics: Dict[str, str] = {}
    for path in _TOPIC_PATHS[split]:
        if not path.exists():
            raise FileNotFoundError(f"Topic file not found: {path}")
        tree = ET.parse(str(path))
        for topic in tree.getroot():
            num = topic.get("number", "")
            if not num:
                continue
            title_el = topic.find("Title")
            question_el = topic.find("Question")
            title = _html_to_text_with_math(title_el.text or "") if title_el is not None else ""
            question = _html_to_text_with_math(question_el.text or "") if question_el is not None else ""
            query = f"{title} {question}".strip()
            if query:
                topics[num] = latex_to_math_format(query)
    return topics


def load_qrels(split: str) -> Dict[str, Dict[str, float]]:
    """Load Task 1 qrels.  Returns {topic_id: {post_id: grade}}."""
    qrels: Dict[str, Dict[str, float]] = {}
    for path in _QREL_PATHS[split]:
        if not path.exists():
            raise FileNotFoundError(f"Qrel file not found: {path}")
        with open(path) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 4:
                    continue
                topic, _, post_id, grade = parts[0], parts[1], parts[2], parts[3]
                try:
                    qrels.setdefault(topic, {})[post_id] = float(grade)
                except ValueError:
                    continue
    return qrels


# ---------------------------------------------------------------------------
# Post corpus — XML (preferred) with JSONL fallback
# ---------------------------------------------------------------------------

_POST_TYPE_MAP = {"answer": "2", "question": "1"}


def _iter_posts_xml(
    post_ids: Optional[set] = None,
    post_type: Optional[str] = "answer",
) -> Iterator[Tuple[str, str]]:
    """
    Iterate the raw Posts XML, yielding (post_id, text) with formulas inline.
    Supports early termination when all requested post_ids are found.
    """
    pt_filter = _POST_TYPE_MAP.get(post_type) if post_type else None
    found: set = set()
    for _, elem in ET.iterparse(str(_POSTS_XML), events=("end",)):
        if elem.tag != "row":
            elem.clear()
            continue
        if pt_filter and elem.get("PostTypeId") != pt_filter:
            elem.clear()
            continue
        pid = elem.get("Id", "")
        if post_ids and pid not in post_ids:
            elem.clear()
            continue
        body = elem.get("Body", "")
        text = _html_to_text_with_math(body)
        if text:
            yield pid, text
            if post_ids is not None:
                found.add(pid)
                if len(found) == len(post_ids):
                    elem.clear()
                    break
        elem.clear()


def _iter_posts_jsonl(
    post_ids: Optional[set] = None,
    post_type: Optional[str] = "answer",
) -> Iterator[Tuple[str, str]]:
    """Fallback: iterate posts.jsonl (formulas appended, not inline)."""
    with open(_POSTS_JSONL) as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if post_type and d.get("post_type") != post_type:
                continue
            pid = str(d.get("post_id", ""))
            if post_ids and pid not in post_ids:
                continue
            title = d.get("title") or ""
            text = d.get("text") or ""
            formulas = d.get("formulas") or []
            formula_str = " ".join(f"[MATH]{f}[/MATH]" for f in formulas if f)
            combined = " ".join(filter(None, [title, text, formula_str])).strip()
            if combined:
                yield pid, latex_to_math_format(combined)


def iter_posts(
    post_ids: Optional[set] = None,
    post_type: Optional[str] = "answer",
) -> Iterator[Tuple[str, str]]:
    """
    Iterate post corpus, yielding (post_id, text).

    Prefers the raw XML (inline formulas preserved) over the processed JSONL
    (formulas stripped and appended, losing sentence coherence).
    """
    if _POSTS_XML.exists():
        yield from _iter_posts_xml(post_ids=post_ids, post_type=post_type)
    elif _POSTS_JSONL.exists():
        print(
            "[WARNING] Raw Posts XML not found — falling back to posts.jsonl. "
            "Formulas will be appended rather than inline.",
            flush=True,
        )
        yield from _iter_posts_jsonl(post_ids=post_ids, post_type=post_type)
    else:
        raise FileNotFoundError(
            f"No post corpus found. Expected:\n  {_POSTS_XML}\n  {_POSTS_JSONL}"
        )


def load_post_texts(
    post_ids: set,
    post_type: Optional[str] = "answer",
) -> Dict[str, str]:
    """Load post_id → text for a set of post IDs."""
    return dict(iter_posts(post_ids=post_ids, post_type=post_type))


def get_judged_post_ids(qrels: Dict[str, Dict[str, float]]) -> set:
    """Extract all post IDs that appear in the qrels."""
    ids = set()
    for cands in qrels.values():
        ids.update(cands.keys())
    return ids

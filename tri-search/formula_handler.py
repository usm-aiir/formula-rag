from __future__ import annotations

"""

Pass full post to llm to pull out important formulas

inverse document frequency, investegate unique formulas, num docs / number of times formula appears

"""
import csv
import html
import json
import logging
import os
import re
import sqlite3
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

try:
    import requests as _requests
except ImportError:
    _requests = None  

# Allow importing formula_utils from this same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from formula_utils import latex_to_mathml, trim_math_delimiters

logger = logging.getLogger(__name__)


def _candidate_roots() -> List[Path]:
    roots: List[Path] = []
    seen: set[Path] = set()
    for start in (Path.cwd().resolve(), Path(__file__).resolve().parent):
        for root in (start, *start.parents):
            if root not in seen:
                seen.add(root)
                roots.append(root)
    return roots


@dataclass(frozen=True)
class TangentCFTConfig:
    python_executable: str
    params: Dict[str, str]
    timeout_seconds: int = 120


class FormulaHandler:
    def __init__(self, formulas: List[str]) -> None:
        self.formulas = formulas
        self._mapping_connection: Optional[sqlite3.Connection] = None
        self.formula_search_root = self._resolve_formula_search_root()
        self.config = self._build_tangent_config()
        self.latefusion_data_root = self.formula_search_root / "LateFusionModel" / "data"
        self.mapping_db_path = self.latefusion_data_root / "tsv_index.sqlite"
        self.posts_xml_path = self.latefusion_data_root / "Posts.V1.3.xml"
        self._question_cache: Dict[int, str] = {}
        self._question_missing: Set[int] = set()

    def __del__(self) -> None:
        if self._mapping_connection is not None:
            try:
                self._mapping_connection.close()
            except Exception:
                pass
            self._mapping_connection = None

    def retrieve_similar_formulas(self, top_k: int = 10) -> List[dict]:
        if top_k <= 0:
            return []

        all_results: List[dict] = []
        for formula_index, curr_formula in enumerate(self.formulas, start=1):
            query_path = self._write_query_tsv(curr_formula, formula_index=formula_index)
            if query_path is None:
                continue

            result_path = query_path.with_suffix(".results.jsonl")
            resolved_result_path: Optional[Path] = None
            try:
                raw_results, resolved_result_path = self._run_tangent_cft(
                    query_path=query_path,
                    result_path=result_path,
                )
                top_hits = raw_results[: max(0, top_k)]
                enriched_hits = self._enrich_hits_with_mapping(top_hits)

                for item in enriched_hits:
                    returned_id = item.get("id")
                    mapped_formula = item.get("formula")
                    all_results.append(
                        {
                            "original_question": item.get("original_question"),
                            "searched_formula": curr_formula,
                            "returned_formula": mapped_formula if mapped_formula else (str(returned_id) if returned_id is not None else None),
                            "result_id": returned_id,
                            "post_id": item.get("post_id"),
                            "thread_id": item.get("thread_id"),
                            "mapping_id": item.get("mapping_id"),
                            "mapping_visual_id": item.get("mapping_visual_id"),
                            "mapping_type": item.get("mapping_type"),
                            "rank": item.get("rank"),
                            "score": item.get("score", 0.0),
                        }
                    )
            finally:
                query_path.unlink(missing_ok=True)
                result_path.unlink(missing_ok=True)
                fallback_result_path = self.formula_search_root / "test_retrieval_results.jsonl"
                if resolved_result_path is not None and resolved_result_path != fallback_result_path:
                    resolved_result_path.unlink(missing_ok=True)

        return all_results

    def _resolve_formula_search_root(self) -> Path:
        env_root = os.getenv("FORMULA_SEARCH_ROOT")
        candidates: List[Path] = []
        if env_root:
            candidates.append(Path(env_root).expanduser())

        for root in _candidate_roots():
            candidates.append(root)
            candidates.append(root / "formula-search")

        seen: set[Path] = set()
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            if (resolved / "tangent_cft_front_end.py").exists():
                return resolved

        raise FileNotFoundError("Could not locate formula-search root with tangent_cft_front_end.py")

    def _build_tangent_config(self) -> TangentCFTConfig:
        latefusion_venv_python = self.formula_search_root / "LateFusionModel" / "venv" / "bin" / "python"
        root_venv_python = self.formula_search_root / "venv" / "bin" / "python"
        if latefusion_venv_python.exists():
            python_executable = str(latefusion_venv_python)
        elif root_venv_python.exists():
            python_executable = str(root_venv_python)
        else:
            python_executable = os.getenv("FORMULA_SEARCH_PYTHON", "python3")

        params = {
            "-ds": "input",
            "-cid": "1",
            "-em": "slt_encoder.tsv",
            "--mp": "slt_model",
            "--wiki": "false",
            "--stream": "true",
            "--faiss": "true",
            "--t": "false",
            "--r": "true",
        }
        return TangentCFTConfig(python_executable=python_executable, params=params)

    def _write_query_tsv(self, formula: str, *, formula_index: int) -> Optional[Path]:
        if not formula or not formula.strip():
            return None

        mathml = latex_to_mathml(trim_math_delimiters(formula))
        if not mathml:
            return None

        tmp = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".tsv",
            delete=False,
            dir=str(self.formula_search_root),
            newline="",
            encoding="utf-8",
        )
        with tmp:
            writer = csv.writer(tmp, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
            writer.writerow(["id", "topic_id", "thread_id", "type", "formula"])
            writer.writerow([f"q_{formula_index}", "mathmex", "mathmex1", "title", mathml])
        return Path(tmp.name)

    def _run_tangent_cft(self, *, query_path: Path, result_path: Path) -> Tuple[List[dict], Optional[Path]]:
        query_rel = self._relative_to_root_or_name(query_path)
        result_rel = self._relative_to_root_or_name(result_path)
        fallback_stream_file = self.formula_search_root / "test_retrieval_results.jsonl"
        fallback_start_offset = fallback_stream_file.stat().st_size if fallback_stream_file.exists() else 0

        command = [self.config.python_executable, "tangent_cft_front_end.py"]
        for key, value in self.config.params.items():
            command.extend([key, value])
        command.extend(["--rf", result_rel, "--qd", query_rel])

        try:
            process = subprocess.run(
                command,
                cwd=str(self.formula_search_root),
                check=True,
                capture_output=True,
                text=True,
                timeout=self.config.timeout_seconds,
            )
            if process.stderr:
                logger.debug("TangentCFT stderr: %s", process.stderr[:1000])
        except Exception as error:
            logger.warning("TangentCFT execution failed for query %s: %s", query_path.name, error)
            return [], None

        result_candidates = [
            result_path,
            self.formula_search_root / result_rel,
            self.formula_search_root / "Retrieval_Results" / result_path.name,
            fallback_stream_file,
        ]
        resolved_result_path = next((path for path in result_candidates if path.exists()), None)
        if resolved_result_path is None:
            logger.warning("TangentCFT finished but no result file found for %s", result_path.name)
            return [], None

        parsed = self._parse_tangent_results(resolved_result_path)
        if not parsed:
            parsed = self._parse_stream_delta_results(
                fallback_file_path=fallback_stream_file,
                start_offset=fallback_start_offset,
            )
            if parsed:
                resolved_result_path = fallback_stream_file

        parsed.sort(key=lambda item: item.get("rank", 0))
        return parsed, resolved_result_path

    def _parse_tangent_results(self, result_path: Path) -> List[dict]:
        parsed: List[dict] = []
        with open(result_path, "r", encoding="utf-8") as file:
            for idx, line in enumerate(file, start=1):
                line = line.strip()
                if not line:
                    continue

                raw_id: Optional[object] = None
                raw_rank: object = idx
                raw_score: object = 0.0

                try:
                    obj = json.loads(line)
                    raw_id = obj.get("id") or obj.get("doc_id") or obj.get("docno") or obj.get("document")
                    raw_rank = obj.get("rank", idx)
                    raw_score = obj.get("score") or obj.get("sim") or 0.0
                except json.JSONDecodeError:
                    parts = line.split()
                    if len(parts) >= 5:
                        raw_id = parts[2]
                        raw_rank = parts[3]
                        raw_score = parts[4]

                parsed_id = self._parse_doc_id(raw_id)
                if parsed_id is None:
                    continue

                try:
                    rank = int(raw_rank)
                except Exception:
                    rank = idx

                try:
                    score = float(raw_score)
                except Exception:
                    score = 0.0

                parsed.append({"id": parsed_id, "rank": rank, "score": score})

        return parsed

    def _parse_stream_delta_results(self, *, fallback_file_path: Path, start_offset: int) -> List[dict]:
        if not fallback_file_path.exists():
            return []

        parsed: List[dict] = []
        with open(fallback_file_path, "r", encoding="utf-8") as file:
            if start_offset > 0:
                file.seek(start_offset)
            for idx, line in enumerate(file, start=1):
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) < 5:
                    continue

                parsed_id = self._parse_doc_id(parts[2])
                if parsed_id is None:
                    continue

                try:
                    rank = int(parts[3])
                except Exception:
                    rank = idx

                try:
                    score = float(parts[4])
                except Exception:
                    score = 0.0

                parsed.append({"id": parsed_id, "rank": rank, "score": score})

        return parsed

    def _parse_doc_id(self, raw_id: Optional[object]) -> Optional[int]:
        if raw_id is None:
            return None
        raw_text = str(raw_id).strip()
        if not raw_text:
            return None
        if ":" in raw_text:
            raw_text = raw_text.rsplit(":", 1)[-1].strip()
        try:
            return int(raw_text)
        except Exception:
            return None

    def _relative_to_root_or_name(self, file_path: Path) -> str:
        try:
            return str(file_path.relative_to(self.formula_search_root))
        except ValueError:
            return file_path.name

    def _get_mapping_connection(self) -> Optional[sqlite3.Connection]:
        if not self.mapping_db_path.exists():
            return None
        if self._mapping_connection is None:
            connection = sqlite3.connect(str(self.mapping_db_path))
            connection.row_factory = sqlite3.Row
            self._mapping_connection = connection
        return self._mapping_connection

    def _lookup_formula_mapping(self, tangent_id: int) -> Dict[str, Optional[object]]:
        connection = self._get_mapping_connection()
        if connection is None:
            return {}

        cursor = connection.cursor()
        cursor.execute(
            """
            SELECT id, post_id, thread_id, type, visual_id, formula
            FROM records
            WHERE id = ? OR visual_id = ?
            ORDER BY CASE WHEN id = ? THEN 0 ELSE 1 END, id ASC
            LIMIT 1
            """,
            (tangent_id, tangent_id, tangent_id),
        )
        row = cursor.fetchone()
        if row is None:
            return {}

        return {
            "mapping_id": self._parse_doc_id(row["id"]),
            "post_id": self._parse_doc_id(row["post_id"]),
            "thread_id": self._parse_doc_id(row["thread_id"]),
            "mapping_type": str(row["type"]).strip() if row["type"] is not None else None,
            "mapping_visual_id": self._parse_doc_id(row["visual_id"]),
            "formula": str(row["formula"]).strip() if row["formula"] is not None else None,
        }

    def _enrich_hits_with_mapping(self, raw_hits: List[dict]) -> List[dict]:
        enriched: List[dict] = []
        question_ids: Set[int] = set()

        for hit in raw_hits:
            tangent_id = hit.get("id")
            if tangent_id is None:
                enriched.append(hit)
                continue

            mapping = self._lookup_formula_mapping(int(tangent_id))
            enriched_hit = dict(hit)
            enriched_hit.update(mapping)
            thread_id = mapping.get("thread_id")
            if isinstance(thread_id, int):
                question_ids.add(thread_id)
            enriched.append(enriched_hit)

        self._load_questions_for_ids(question_ids)

        for hit in enriched:
            thread_id = hit.get("thread_id")
            if isinstance(thread_id, int):
                question_text = self._question_cache.get(thread_id)
                hit["original_question"] = question_text if question_text else f"Question #{thread_id}"
            else:
                hit["original_question"] = None

        return enriched

    def _load_questions_for_ids(self, question_ids: Set[int]) -> None:
        if not question_ids:
            return

        missing = {qid for qid in question_ids if qid not in self._question_cache and qid not in self._question_missing}
        if not missing:
            return

        # Primary source: local Posts.V1.3.xml (ARQMath dataset dump)
        if self.posts_xml_path.exists():
            try:
                context = ET.iterparse(str(self.posts_xml_path), events=("end",))
                for _, elem in context:
                    if elem.tag != "row":
                        continue

                    attrs = elem.attrib
                    post_type = attrs.get("PostTypeId")
                    post_id = self._parse_doc_id(attrs.get("Id"))
                    if post_type == "1" and isinstance(post_id, int) and post_id in missing:
                        self._question_cache[post_id] = self._build_question_text(attrs)
                        missing.remove(post_id)
                        if not missing:
                            elem.clear()
                            break

                    elem.clear()
            except Exception as error:
                logger.warning("Failed loading question text from %s: %s", self.posts_xml_path, error)
        else:
            logger.debug("Posts.V1.3.xml not found at %s, falling back to Stack Exchange API", self.posts_xml_path)

        # Fallback: Math Stack Exchange API for any IDs not found in the XML
        if missing:
            self._fetch_questions_from_mse_api(missing)

    def _fetch_questions_from_mse_api(self, question_ids: Set[int]) -> None:
        """Fetch question titles from the Math Stack Exchange API."""
        if _requests is None:
            logger.warning("requests library not available; cannot fetch question titles from MSE API")
            self._question_missing.update(question_ids)
            return

        # API allows up to 100 IDs per request
        ids_list = list(question_ids)
        batch_size = 100
        for i in range(0, len(ids_list), batch_size):
            batch = ids_list[i : i + batch_size]
            ids_param = ";".join(str(qid) for qid in batch)
            url = f"https://api.stackexchange.com/2.3/questions/{ids_param}"
            params = {"site": "math", "filter": "withbody", "pagesize": batch_size}
            try:
                response = _requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                for item in data.get("items", []):
                    qid = item.get("question_id")
                    title = item.get("title", "").strip()
                    if qid and title:
                        # Unescape HTML entities in the title (MSE API returns HTML-encoded titles)
                        self._question_cache[int(qid)] = html.unescape(title)
                # Mark any IDs not returned by the API as missing
                returned_ids = {item.get("question_id") for item in data.get("items", [])}
                for qid in batch:
                    if qid not in returned_ids:
                        self._question_missing.add(qid)
            except Exception as error:
                logger.warning("Math Stack Exchange API request failed: %s", error)
                self._question_missing.update(batch)

    def _build_question_text(self, attrs: Dict[str, str]) -> str:
        title = (attrs.get("Title") or "").strip()
        if title:
            return title

        body = attrs.get("Body") or ""
        body_text = re.sub(r"<[^>]+>", " ", body)
        body_text = html.unescape(re.sub(r"\s+", " ", body_text)).strip()
        if len(body_text) > 180:
            return body_text[:177] + "..."
        return body_text


def _scrape_mse_post_text(thread_id: int, max_chars: int = 1500) -> str:
    """Scrape the question and top answers text from a Math Stack Exchange post."""
    if _requests is None:
        return ""
    try:
        from bs4 import BeautifulSoup
        url = f"https://math.stackexchange.com/q/{thread_id}"
        headers = {"User-Agent": "MathMexFormulaHandler/0.1"}
        resp = _requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        parts: List[str] = []

        # Question body
        question_body = soup.select_one(".question .s-prose")
        if question_body:
            parts.append("Question:\n" + question_body.get_text(separator=" ", strip=True))

        # Top answers (up to 2)
        for answer in soup.select(".answer .s-prose")[:2]:
            parts.append("Answer:\n" + answer.get_text(separator=" ", strip=True))

        text = "\n\n".join(parts).strip()
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
        return text
    except Exception as err:
        return f"(could not fetch post text: {err})"


if __name__ == "__main__":
    import json

    example_formulas = [
        r"\frac{d}{dx} e^x",
        r"\sum_{n=0}^{\infty} \frac{x^n}{n!}",
    ]
    print(f"Searching for {len(example_formulas)} formula(s):\n")
    for f in example_formulas:
        print(f"  {f}")
    print()

    handler = FormulaHandler(example_formulas)
    print(f"TangentCFT root: {handler.formula_search_root}\n")

    results = handler.retrieve_similar_formulas(top_k=5)

    if not results:
        print("No results returned (TangentCFT may not be running or no index loaded).")
    else:
        print(f"{len(results)} result(s):\n")
        for i, r in enumerate(results, start=1):
            thread_id = r['thread_id']
            print(f"[{i}] rank={r['rank']}  score={r['score']:.4f}")
            print(f"     searched : {r['searched_formula']}")
            print(f"     returned : {r['returned_formula']}")
            print(f"     question : {r['original_question']}")
            print(f"     post_id  : {r['post_id']}  thread_id={thread_id}")
            if thread_id:
                post_text = _scrape_mse_post_text(thread_id)
                if post_text:
                    print(f"     post text:\n{post_text}")
            print()

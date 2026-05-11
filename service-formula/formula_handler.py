from __future__ import annotations
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
from formula_utils import latex_to_mathml, trim_math_delimiters, scrape_mse_post_text

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class TangentCFTConfig:
    python_executable: str
    # CLI flags forwarded verbatim to tangent_cft_front_end.py.
    # Example: {"-ds": "input", "--faiss": "true", "--t": "false", ...}
    params: Dict[str, str]
    timeout_seconds: int = 120
    
def _candidate_roots() -> List[Path]:
    """ Generate candidate root directories to search for the formula-search codebase, 
    starting from the current file and the current working directory, and including their
    parent directories. This allows flexibility in where the code is run from while still
    being able to locate the necessary files for TangentCFT."""
    roots: List[Path] = []
    seen: set[Path] = set()
    for start in (Path.cwd().resolve(), Path(__file__).resolve().parent):
        for root in (start, *start.parents):
            if root not in seen:
                seen.add(root)
                roots.append(root)
    return roots


class FormulaHandler:
    def __init__(self, formulas: List[str]) -> None:
        self.formulas = formulas
        self._mapping_connection: Optional[sqlite3.Connection] = None
        self.formula_search_root = self._resolve_formula_search_root()
        self.config = self._build_tangent_config()
        self.mapping_db_path = self.formula_search_root / "data" / "tsv_index.sqlite"
        self.posts_xml_path = self.formula_search_root.parent / "ARQMathQueries" / "Posts.V1.3.xml"
        self._question_cache: Dict[int, str] = {}
        self._question_missing: Set[int] = set()

    def __del__(self) -> None:
        """Ensure any open database connections are closed when the handler is destroyed."""
        if self._mapping_connection is not None:
            try:
                self._mapping_connection.close()
            except Exception:
                pass
            self._mapping_connection = None

    def retrieve_similar_formulas(self, top_k: int = 10) -> List[dict]:
        """Run TangentCFT retrieval for every formula in self.formulas and return enriched results.

        Each result dict contains:
          {
            "original_question": "Integration by parts",   # title of the MSE thread
            "searched_formula":  r"\\int u dv",            # the LaTeX we queried with
            "returned_formula":  r"\\int_a^b f(x) dx",     # LaTeX of the matched formula
            "result_id":         48291,                    # TangentCFT internal integer ID
            "post_id":           102934,                   # ARQMath/MSE post ID
            "thread_id":         98123,                    # parent question post ID
            "mapping_id":        48291,                    # SQLite records.id (usually same as result_id)
            "mapping_visual_id": 7731,                     # ARQMath visual formula ID
            "mapping_type":      "answer",                 # post type from the index
            "rank":              1,
            "score":             0.9312,
          }
        """
        # Clean up any .tsv / .results.jsonl files left in formula-search/ by previous
        # runs that crashed before the finally-block cleanup could execute.
        self._cleanup_stray_temp_files()

        all_results: List[dict] = []

        for formula_index, curr_formula in enumerate(self.formulas, start=1):
            query_path = self._write_query_tsv(curr_formula, formula_index=formula_index)
            # error handling, if the query file didnt get created everything with fail, so skip it and move on
            if query_path is None:
                continue

            result_path = query_path.with_suffix(".results.jsonl")
            resolved_result_path: Optional[Path] = None
            try:
                raw_results, resolved_result_path = self._run_tangent_cft(
                    query_path=query_path,
                    result_path=result_path,
                )
                top_hits = raw_results[:max(0, top_k)]
                enriched_hits = self._map_formulas_with_question_info(top_hits)

                for item in enriched_hits:
                    returned_id = item.get("id")
                    # Prefer the LaTeX string from the SQLite mapping; fall back to the raw
                    # integer ID as a string if the mapping lookup found nothing.
                    mapped_formula = item.get("formula")
                    returned_formula = mapped_formula or (str(returned_id) if returned_id is not None else None)
                    all_results.append(
                        {
                            "original_question": item.get("original_question"),
                            "searched_formula": curr_formula,
                            "returned_formula": returned_formula,
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
                # Always delete the query TSV we wrote and the expected result path.
                # resolved_result_path is also deleted unless it is the persistent
                # streaming file (test_retrieval_results.jsonl), which TangentCFT owns.
                query_path.unlink(missing_ok=True)
                result_path.unlink(missing_ok=True)
                fallback_result_path = self.formula_search_root / "test_retrieval_results.jsonl"
                if resolved_result_path is not None and resolved_result_path != fallback_result_path:
                    resolved_result_path.unlink(missing_ok=True)

        return all_results

    def _cleanup_stray_temp_files(self) -> None:
        """Delete any tmp*.tsv or tmp*.results.jsonl files left in the formula-search
        directory by a previous run that crashed before its finally-block could execute."""
        for pattern in ("tmp*.tsv", "tmp*.results.jsonl"):
            for stray in self.formula_search_root.glob(pattern):
                try:
                    stray.unlink()
                    logger.debug("Cleaned up stray temp file: %s", stray.name)
                except OSError as err:
                    logger.warning("Could not remove stray temp file %s: %s", stray.name, err)

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
        # Candidate Python executables in priority order:
        #   1. formula-search/LateFusionModel/venv  — dedicated venv shipped with TangentCFT
        #   2. formula-search/venv                  — repo-level venv
        #   3. ~/.venv                              — project-wide venv (has faiss installed)
        #   4. FORMULA_SEARCH_PYTHON env var        — user-specified override
        #   5. system python3                       — last resort (may be missing faiss)
        latefusion_venv_python = self.formula_search_root / "LateFusionModel" / "venv" / "bin" / "python"
        root_venv_python = self.formula_search_root / "venv" / "bin" / "python"
        project_venv_python = Path.home() / ".venv" / "bin" / "python"
        if latefusion_venv_python.exists():
            python_executable = str(latefusion_venv_python)
        elif root_venv_python.exists():
            python_executable = str(root_venv_python)
        elif project_venv_python.exists():
            python_executable = str(project_venv_python)
        else:
            python_executable = os.getenv("FORMULA_SEARCH_PYTHON", "python3")

        # CLI parameters for tangent_cft_front_end.py:
        #   -ds      data source directory inside formula-search/ (contains the pre-encoded corpus)
        #   -cid     configuration file ID → loads Configuration/config/config_1
        #   -em      encoder map file → Embedding_Preprocessing/slt_encoder.tsv (token→int vocabulary)
        #   --mp     pre-trained FastText model path prefix → slt_model.wv.vectors.npy
        #   --wiki   False = use MSE streaming dataset format (not Wikipedia)
        #   --stream True  = load corpus from encoded.jsonl instead of re-encoding on startup
        #   --faiss  True  = use FAISS ANN index instead of brute-force cosine similarity
        #   --t      False = skip training (model already trained, load from --mp)
        #   --r      True  = run retrieval on the provided query file
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
        # Record how many bytes are already in the streaming output file before we launch
        # TangentCFT. If the dedicated result file is missing after the run, we read only
        # the lines appended beyond this offset (the delta from this single query).
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
        except subprocess.CalledProcessError as error:
            # Log the full stderr so the root cause (e.g. missing module) is visible.
            stderr_snippet = (error.stderr or "")[:2000]
            logger.warning(
                "TangentCFT returned exit code %d for query %s.\nstderr:\n%s",
                error.returncode, query_path.name, stderr_snippet,
            )
            return [], None
        except Exception as error:
            logger.warning("TangentCFT execution failed for query %s: %s", query_path.name, error)
            return [], None

        # result_path is already absolute inside formula_search_root, so
        # formula_search_root / result_rel would resolve to the same path — skip it.
        result_candidates = [
            result_path,
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
        """Parse a TangentCFT result file which may be in one of two formats:

        JSON lines (streaming mode):
          {"id": "formula:48291", "rank": 1, "score": 0.931}

        TREC format (plain text, space-separated, 6+ columns):
          topic_id  Q0  doc_id  rank  score  run_name
          mathmex   Q0  48291   1     0.931  TangentCFT
        """
        parsed: List[dict] = []
        with open(result_path, "r", encoding="utf-8") as file:
            for idx, line in enumerate(file, start=1):
                line = line.strip()
                if not line:
                    continue

                raw_id: object = None
                raw_rank: object = idx
                raw_score: object = 0.0

                try:
                    obj = json.loads(line)
                    # TangentCFT uses different key names depending on version/config.
                    raw_id = obj.get("id") or obj.get("doc_id") or obj.get("docno") or obj.get("document")
                    raw_rank = obj.get("rank", idx)
                    raw_score = obj.get("score") or obj.get("sim") or 0.0
                except json.JSONDecodeError:
                    # Fall back to TREC column layout: [topic, Q0, doc_id, rank, score, run]
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
        """Read only the lines appended to the streaming output file since start_offset.

        TangentCFT appends TREC-format lines to test_retrieval_results.jsonl during streaming:
          topic_id  Q0  doc_id  rank  score  run_name
          mathmex   Q0  48291   1     0.931  TangentCFT
        """
        if not fallback_file_path.exists():
            return []

        parsed: List[dict] = []
        with open(fallback_file_path, "r", encoding="utf-8") as file:
            # Seek past lines that existed before this query was issued.
            if start_offset > 0:
                file.seek(start_offset)
            for idx, line in enumerate(file, start=1):
                line = line.strip()
                if not line:
                    continue

                # TREC columns: [topic_id, Q0, doc_id, rank, score, run_name]
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

    def _parse_doc_id(self, raw_id: object) -> Optional[int]:
        """Parse a TangentCFT document ID to a plain integer.

        TangentCFT sometimes emits IDs in a namespaced format such as
        "formula:48291" or "thread:98123". We only need the numeric part.
        """
        if raw_id is None:
            return None
        raw_text = str(raw_id).strip()
        if not raw_text:
            return None
        # Strip the namespace prefix (e.g. "formula:48291" → "48291").
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
        # Prefer an exact id match over a visual_id match; the CASE WHEN gives id=? a
        # sort weight of 0 (higher priority) and visual_id=? a weight of 1.
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

    def _map_formulas_with_question_info(self, raw_hits: List[dict]) -> List[dict]:
        """Enrich raw TangentCFT hits with SQLite mapping data and question titles.

        raw_hits come in as: [{"id": 48291, "rank": 1, "score": 0.931}, ...]
        Each hit is augmented in-place with the fields from _lookup_formula_mapping:
          post_id, thread_id, mapping_visual_id, mapping_type, formula (LaTeX string)
        Then original_question is added by loading the question title for each thread_id.
        """
        enriched: List[dict] = []
        question_ids: Set[int] = set()

        for hit in raw_hits:
            tangent_id = hit.get("id")

            # The ID can be None when TangentCFT emits a malformed result line that
            # _parse_tangent_results could not extract a numeric ID from.
            if tangent_id is None:
                enriched.append(hit)
                continue

            # Look up the ARQMath post_id, thread_id, and LaTeX string for this
            # TangentCFT-internal integer ID via the tsv_index.sqlite bridge table.
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
        """Load question titles for the given set of question IDs, using a combination of local XML parsing and API calls."""
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
            # No custom filter needed — the default filter includes title, which is all we use.
            # "withbody" would add the full HTML body to every response, bloating it unnecessarily.
            params = {"site": "math", "pagesize": batch_size}
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
                post_text = scrape_mse_post_text(thread_id)
                if post_text:
                    print(f"     post text:\n{post_text}")
            print()

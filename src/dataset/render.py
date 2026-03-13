"""
Render mathematical formulas from the formula index as PNG images.

Two image types are produced per unique formula (identified by old_visual_id):

  {id}_typeset.png  — The LaTeX rendered as typeset mathematics using
                      matplotlib's mathtext engine.  Only saved if matplotlib
                      can actually parse the formula (not a text fallback), and
                      only for formulas with meaningful visual complexity.

  {id}_graph.png    — A plotted graph of the formula when it can be parsed
                      as a single-variable function by sympy.  Requires the
                      `antlr4-python3-runtime` package (already in requirements).

Quality filters (applied during render AND prune):

  - Trivial formulas: stripped LaTeX ≤ 2 chars or matches a single-token
    pattern (e.g. just "a", "M", "x_1") → skipped entirely.

  - Unparseable by mathtext: formulas containing LaTeX environments
    (\begin{...}) or macros (\color{}) that matplotlib cannot typeset are
    detected via MathTextParser pre-validation and dropped — these would
    otherwise render as raw LaTeX source text, which is not useful visual data.

  - Graph candidates: only attempt graph rendering for formulas that look like
    plottable single-variable expressions (contain x/y and math operators),
    and reject sympy Eq/Relational objects (equations, not functions).

Priority
--------
By default, only formulas belonging to "priority posts" are rendered:
  - Posts judged in ARQMath Task 1 qrels (answer retrieval training/eval)
  - Posts with downloaded images (trimodal alignment training)

Run with --all to render the full corpus (background job, 28M formulas).

Modes
-----
    python -m src.dataset.render_formulas               # priority posts
    python -m src.dataset.render_formulas --all         # full corpus
    python -m src.dataset.render_formulas --prune       # delete bad renders
    python -m src.dataset.render_formulas --prune --dry-run  # preview only
    python -m src.dataset.render_formulas --regraph     # retry graph renders

Output
------
    data/processed/rendered_formulas/{old_visual_id}_typeset.png
    data/processed/rendered_formulas/{old_visual_id}_graph.png
"""

from __future__ import annotations

import argparse
import os
import re
import signal
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pyarrow.parquet as pq
from tqdm import tqdm

REPO_ROOT         = Path(__file__).resolve().parents[2]
FORMULA_INDEX_DIR = REPO_ROOT / "data/processed/formula_index"
RENDERED_DIR      = REPO_ROOT / "data/processed/rendered_formulas"
IMAGES_DIR        = REPO_ROOT / "data/raw/arqmath/images"
QRELS_DIR         = REPO_ROOT / "data/raw/arqmath/qrels"

DPI             = 150
TYPESET_FIGSIZE = (8, 1.5)
GRAPH_FIGSIZE   = (6, 4)
GRAPH_TIMEOUT   = 5      # seconds per formula for graph generation

# Minimum stripped LaTeX length to produce a typeset render
MIN_LATEX_LEN = 3

# Fast regex rejects: LaTeX features matplotlib mathtext cannot typeset.
# These are checked BEFORE calling the parser (sub-microsecond per formula).
#   \begin{...}  — environments (array, pmatrix, aligned, etc.)
#   \color{      — color macros
#   \text{       — text-mode inside math (causes ValueError in ~40% of failures)
_FAST_REJECT_RE = re.compile(r'\\begin\{|\\color\{|\\text\{')

# Bare x/y variable (not preceded by a backslash or letter — avoids \xi, xy)
_VAR_RE  = re.compile(r'(?<![a-zA-Z\\])[xy](?![a-zA-Z_{])')
# Formula must have at least one operator or named function to be plottable
_EXPR_RE = re.compile(r'(?:sin|cos|tan|exp|log|sqrt|\^|\+|\\frac|\\sqrt)')

# Module-level MathTextParser singleton — lazy-initialised once per process.
# Avoids the ~100 ms constructor overhead when checking thousands of formulas.
_mpl_parser = None


def _get_mpl_parser():
    global _mpl_parser
    if _mpl_parser is None:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib.mathtext import MathTextParser
        _mpl_parser = MathTextParser("Agg")
    return _mpl_parser


# ── Quality filters ───────────────────────────────────────────────────────────

# Spacing commands that add no visual content
_STRIP_SPACING_RE  = re.compile(r'\\[,;:!\s]|\\ ')
# Font / style declarations that wrap a single character
_STRIP_FONT_RE     = re.compile(
    r'\\(?:rm|bf|it|sf|tt|sc|displaystyle|textstyle|scriptstyle)\s*'
    r'|\\math(?:bf|rm|it|sf|tt|bb|cal|frak|scr)\{([^}]{1,3})\}'
)


def _trivial_core(s: str) -> str:
    """Strip spacing macros and font/style wrappers to find the bare core."""
    # Remove spacing commands: \, \; \: \! \  \\
    s = _STRIP_SPACING_RE.sub("", s).strip()
    # Replace \math*{X} with just X (font wrapper around 1-3 chars)
    s = re.sub(r'\\math(?:bf|rm|it|sf|tt|bb|cal|frak|scr)\{([^}]{1,3})\}', r'\1', s)
    # Remove bare font declarations (e.g. \rm, \bf before a char)
    s = re.sub(r'\\(?:rm|bf|it|sf|tt|sc|displaystyle|textstyle|scriptstyle)\s*', '', s)
    return s.strip('{}').strip()


# All Greek letter command names (lower + upper)
_GREEK_RE = re.compile(
    r'\\(?:alpha|beta|gamma|delta|epsilon|varepsilon|zeta|eta|theta|vartheta'
    r'|iota|kappa|lambda|mu|nu|xi|pi|varpi|rho|varrho|sigma|varsigma|tau'
    r'|upsilon|phi|varphi|chi|psi|omega'
    r'|Gamma|Delta|Theta|Lambda|Xi|Pi|Sigma|Upsilon|Phi|Psi|Omega)$'
)
# Special math constants that are a single visual object
_SPECIAL_RE = re.compile(r'\\(?:infty|emptyset|nabla|ell|aleph|beth|partial|wp)$')

# Accent wrappers that decorate a single token: \hat{x}, \vec{v}, \overline{z}
_ACCENT_RE = re.compile(
    r'\\(?:hat|vec|bar|tilde|dot|ddot|acute|grave|breve|check|mathring'
    r'|overline|underline|widetilde|widehat)\{([^}]{1,5})\}'
)


def _strip_decorators(s: str) -> str:
    """Remove superscripts, subscripts, primes, accents, and bar-wrappers,
    leaving only the base identifier."""
    # Accent wrappers: \hat{x} → x
    s = _ACCENT_RE.sub(r'\1', s)
    # Superscripts: ^{...} or ^char
    s = re.sub(r'\^\{[^{}]*\}', '', s)
    s = re.sub(r'\^[a-zA-Z0-9*\'+\\-]', '', s)
    # Subscripts: _{...} or _char
    s = re.sub(r'_\{[^{}]*\}', '', s)
    s = re.sub(r'_[a-zA-Z0-9]', '', s)
    # Primes and asterisks
    s = re.sub(r"['*]+", '', s)
    # Absolute-value / norm bars: |x| or \left|x\right|
    s = re.sub(r'\\left\s*\\?\|(.+?)\\right\s*\\?\|', r'\1', s)
    s = re.sub(r'\|([^|]{1,10})\|', r'\1', s)
    return s.strip().strip('{}').strip()


def _is_trivial(latex: str) -> bool:
    """Return True for formulas too simple to produce useful training images.

    Catches:
      - Bare single characters ('a', 'M', '0'), including formatted variants
        like '\\rm B', '\\:0\\:', '\\mathbf{x}'.
      - Single Greek letters: '\\phi', '\\alpha', '\\Omega', etc.
      - Single special symbols: '\\infty', '\\nabla', '\\partial', etc.
      - Single-object expressions with decorators: 'f^{-1}', '\\hat{x}',
        'T^*', 'A^T', '\\vec{v}', '\\overline{z}'.

    Keeps anything that contains at least two distinct mathematical objects
    or a structural/binary relationship between them.
    """
    s = latex.strip()
    if len(s) < MIN_LATEX_LEN:
        return True

    # Fast path: raw single-token (optional subscript/superscript)
    if re.fullmatch(r'[a-zA-Z0-9](?:[_^]\{?[a-zA-Z0-9]\}?)?', s):
        return True

    # Strip formatting/spacing wrappers
    core = _trivial_core(s)
    if len(core) < MIN_LATEX_LEN:
        return True
    if re.fullmatch(r'[a-zA-Z0-9](?:[_^]\{?[a-zA-Z0-9]\}?)?', core):
        return True

    # Strip ALL decorators from the original (not core, which has mangled braces)
    # and check if what remains is a single mathematical object.
    bare = _trivial_core(_strip_decorators(s))
    if not bare:
        return True
    if _GREEK_RE.match(bare) or _SPECIAL_RE.match(bare):
        return True
    if re.fullmatch(r'[a-zA-Z]', bare):
        return True

    return False


def _mathtext_parseable(latex: str) -> bool:
    """Return True only if matplotlib's mathtext can parse this formula.

    Formulas that fail are silently rendered by matplotlib as raw text strings
    — not useful as visual data.  Fast-rejects common known-bad patterns via
    regex before calling the (more expensive) parser.
    """
    if _FAST_REJECT_RE.search(latex):
        return False
    try:
        _get_mpl_parser().parse(f"${latex}$", dpi=72)
        return True
    except Exception:
        return False


def _has_graph_potential(latex: str) -> bool:
    """Quick heuristic: does this formula look like a plottable function?

    Avoids calling sympy for every formula — only forwards likely candidates.
    """
    if len(latex) > 200:
        return False
    # Must have a free variable we can plot against
    if not _VAR_RE.search(latex):
        return False
    # Must have at least one mathematical operator or function
    if not _EXPR_RE.search(latex):
        return False
    return True


# ── Worker helpers (must be top-level for pickling) ───────────────────────────

def _init_worker() -> None:
    """Initialise matplotlib in non-interactive mode for each worker process."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.ioff()


def _render_typeset(formula_id: str, latex: str, out_dir: Path) -> bool:
    """Render LaTeX as a typeset PNG.

    Returns False (without writing) if the formula fails any quality gate:
      - Too trivial (single character / very short)
      - matplotlib mathtext cannot parse it (would render as raw source text)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if _is_trivial(latex):
        return False
    if not _mathtext_parseable(latex):
        return False

    out_path = out_dir / f"{formula_id}_typeset.png"
    if out_path.exists():
        return True

    try:
        fig = plt.figure(figsize=TYPESET_FIGSIZE, dpi=DPI)
        fig.patch.set_facecolor("white")
        ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])
        ax.set_facecolor("white")
        ax.axis("off")
        ax.text(
            0.5, 0.5,
            f"${latex}$",
            ha="center", va="center",
            fontsize=22, color="black",
            transform=ax.transAxes,
        )
        fig.savefig(
            str(out_path),
            bbox_inches="tight",
            dpi=DPI,
            facecolor="white",
            edgecolor="none",
            format="png",
        )
        plt.close(fig)
        return True
    except Exception:
        try:
            plt.close("all")
        except Exception:
            pass
        return False


def _render_graph(formula_id: str, latex: str, out_dir: Path) -> bool:
    """Attempt to render a function graph via sympy + matplotlib.

    Returns True if a graph was produced, False otherwise.
    Requires antlr4-python3-runtime for sympy.parsing.latex.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    if not _has_graph_potential(latex):
        return False

    out_path = out_dir / f"{formula_id}_graph.png"
    if out_path.exists():
        return True

    # Guard: sympy.parsing.latex needs antlr4-python3-runtime
    try:
        from sympy.parsing.latex import parse_latex
        from sympy import lambdify, Eq
        from sympy.core.relational import Relational
    except ImportError:
        return False

    try:
        expr = parse_latex(latex)

        # Equations like x^2 = 0 produce Eq objects — not functions to plot
        if isinstance(expr, (Eq, Relational)):
            return False

        free = expr.free_symbols
        if len(free) != 1:
            return False
        if expr.count_ops() > 60:
            return False

        (x,) = free
        f = lambdify(x, expr, modules=["numpy"])

        x_vals = np.linspace(-10, 10, 500)
        with np.errstate(all="ignore"):
            y_vals = np.array(f(x_vals), dtype=float)

        mask = np.isfinite(y_vals) & (np.abs(y_vals) < 1e6)
        if mask.sum() < 20:
            return False

        fig, ax = plt.subplots(figsize=GRAPH_FIGSIZE, dpi=DPI)
        ax.plot(x_vals[mask], y_vals[mask], linewidth=1.5)
        ax.axhline(y=0, color="k", linewidth=0.5)
        ax.axvline(x=0, color="k", linewidth=0.5)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel(str(x))
        fig.tight_layout()
        fig.savefig(
            str(out_path),
            bbox_inches="tight",
            dpi=DPI,
            facecolor="white",
            edgecolor="none",
            format="png",
        )
        plt.close(fig)
        return True

    except Exception:
        try:
            plt.close("all")
        except Exception:
            pass
        return False


def _render_formula_worker(args: tuple) -> tuple:
    """Worker entry point: render typeset + graph for one formula.

    Uses SIGALRM to enforce a per-formula timeout on the graph step.
    Returns (formula_id, typeset_ok, graph_ok).
    """
    formula_id, latex, out_dir_str, do_graph = args
    out_dir = Path(out_dir_str)

    typeset_ok = _render_typeset(formula_id, latex, out_dir)

    graph_ok = False
    if do_graph:
        if hasattr(signal, "SIGALRM"):
            def _timeout_handler(signum, frame):
                raise TimeoutError("graph render timed out")

            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(GRAPH_TIMEOUT)
            try:
                graph_ok = _render_graph(formula_id, latex, out_dir)
            except (TimeoutError, Exception):
                graph_ok = False
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
        else:
            try:
                graph_ok = _render_graph(formula_id, latex, out_dir)
            except Exception:
                graph_ok = False

    return formula_id, typeset_ok, graph_ok


# ── Priority post loading ─────────────────────────────────────────────────────

def load_priority_post_ids() -> set:
    """Collect post IDs from Task 1 qrels and image post directories."""
    post_ids = set()

    task1_qrels = QRELS_DIR / "task1"
    if task1_qrels.is_dir():
        for qrel_file in task1_qrels.rglob("*"):
            if not qrel_file.is_file():
                continue
            try:
                with qrel_file.open() as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 3:
                            try:
                                post_ids.add(int(parts[2]))
                            except ValueError:
                                pass
            except Exception:
                pass

    if IMAGES_DIR.is_dir():
        for post_dir in IMAGES_DIR.iterdir():
            if post_dir.is_dir():
                try:
                    post_ids.add(int(post_dir.name))
                except ValueError:
                    pass

    return post_ids


# ── Formula ID collection ─────────────────────────────────────────────────────

def collect_formula_ids(priority_post_ids) -> dict:
    """Return a mapping of old_visual_id → latex for formulas to render."""
    if not FORMULA_INDEX_DIR.exists():
        print(f"ERROR: Formula index not found at {FORMULA_INDEX_DIR}")
        sys.exit(1)

    shards = sorted(FORMULA_INDEX_DIR.glob("shard_*.parquet"))
    if not shards:
        print(f"ERROR: No Parquet shards found in {FORMULA_INDEX_DIR}")
        sys.exit(1)

    formula_map = {}

    for shard in tqdm(shards, desc="Scanning formula index", unit="shard"):
        table = pq.read_table(
            shard,
            columns=["post_id", "old_visual_id", "latex", "type"],
        )
        for batch in table.to_batches():
            for post_id, vid, latex, ftype in zip(
                batch.column("post_id").to_pylist(),
                batch.column("old_visual_id").to_pylist(),
                batch.column("latex").to_pylist(),
                batch.column("type").to_pylist(),
            ):
                if ftype == "comment":
                    continue
                if priority_post_ids is not None and post_id not in priority_post_ids:
                    continue
                if vid not in formula_map and latex:
                    formula_map[vid] = latex

    return formula_map


# ── Prune mode ────────────────────────────────────────────────────────────────

def _load_vid_latex_map(target_vids: set) -> dict:
    """Load old_visual_id → latex for a specific set of IDs from the index."""
    shards = sorted(FORMULA_INDEX_DIR.glob("shard_*.parquet"))
    formula_map = {}

    for shard in tqdm(shards, desc="Loading formula index", unit="shard"):
        table = pq.read_table(shard, columns=["old_visual_id", "latex"])
        for vid, latex in zip(
            table.column("old_visual_id").to_pylist(),
            table.column("latex").to_pylist(),
        ):
            if vid in target_vids and vid not in formula_map:
                formula_map[vid] = latex or ""

        if len(formula_map) >= len(target_vids):
            break   # found everything we need

    return formula_map


def _regex_parseable(latex: str) -> bool:
    """Fast regex-only quality check — no MathTextParser call.

    Catches the most common failure modes (environments, color macros, text
    mode) in microseconds per formula.  Used by `prune_bad_renders` for
    speed; the full `_mathtext_parseable` (with the actual parser) is used
    during rendering to prevent any new bad renders from being saved.
    """
    return not _FAST_REJECT_RE.search(latex)


def prune_bad_renders(dry_run: bool = False, n_workers: int = None) -> None:
    """Delete typeset renders that fail quality checks.

    Removes:
      - Trivially simple formulas (single chars, too short)
      - Formulas with patterns that matplotlib mathtext cannot typeset:
        \\begin{...} environments, \\color{}, \\text{} mode  (caught via fast
        regex — the most common failure modes covering ~50% of all bad renders)
      - Orphaned renders whose formula_id is not in the index

    Note: subtle mathtext failures not caught by the fast regex are left in
    place.  They represent imperfect but non-broken renders (not raw source
    text), and are far less common (<3% of the total corpus).  Going forward,
    `_render_typeset` uses the full MathTextParser pre-check so no new bad
    renders will be created.

    Also deletes any companion _graph.png for removed typeset files.
    """
    typeset_pngs = sorted(RENDERED_DIR.glob("*_typeset.png"))
    if not typeset_pngs:
        print("No renders found.")
        return

    n = len(typeset_pngs)
    print(f"Found {n:,} typeset renders to audit.")

    rendered_vids = {p.stem.replace("_typeset", "") for p in typeset_pngs}

    print("\nLoading LaTeX for rendered formulas...")
    vid_latex = _load_vid_latex_map(rendered_vids)

    trivial = unparseable = orphaned = kept = 0
    deleted_list = []

    for png in tqdm(typeset_pngs, desc="Auditing (regex)", unit="img"):
        vid = png.stem.replace("_typeset", "")
        latex = vid_latex.get(vid, "")

        if not latex:
            reason = "orphaned"
            orphaned += 1
        elif _is_trivial(latex):
            reason = "trivial"
            trivial += 1
        elif not _regex_parseable(latex):
            reason = "unparseable"
            unparseable += 1
        else:
            kept += 1
            continue

        deleted_list.append((png, vid))

    total_deleted = trivial + unparseable + orphaned
    print(
        f"\nAudit complete."
        f"\n  Kept          : {kept:,}"
        f"\n  Would delete  : {total_deleted:,}"
        f"\n    Trivial     : {trivial:,}"
        f"\n    Unparseable : {unparseable:,}"
        f"\n    Orphaned    : {orphaned:,}"
    )

    if dry_run:
        print("\n(dry run — no files deleted)")
        return

    print("\nDeleting...")
    for png, vid in tqdm(deleted_list, desc="Deleting", unit="img"):
        png.unlink(missing_ok=True)
        graph = RENDERED_DIR / f"{vid}_graph.png"
        graph.unlink(missing_ok=True)

    print(f"Deleted {total_deleted:,} renders.")


# ── Render runner ─────────────────────────────────────────────────────────────

def _run_parallel_or_serial(tasks: list, desc: str, n_workers: int) -> list:
    """Run _render_formula_worker tasks in parallel, falling back to serial.

    ProcessPoolExecutor requires semaphores which are unavailable in some
    sandboxed environments.  In that case we fall back to single-threaded
    processing so the script still works, just more slowly.
    """
    results = []
    try:
        with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker) as executor:
            futures = {executor.submit(_render_formula_worker, t): t[0] for t in tasks}
            for future in tqdm(as_completed(futures), total=len(futures), desc=desc, unit="formula"):
                try:
                    results.append(future.result(timeout=30))
                except Exception:
                    results.append((None, False, False))
    except Exception as exc:
        if results:
            # Already got some results before failure, continue
            pass
        else:
            # Likely a permission error (sandboxed env) — fall back to serial
            print(f"  (parallel unavailable: {exc}; running single-threaded)")
            _init_worker()
            for task in tqdm(tasks, desc=f"{desc} [serial]", unit="formula"):
                try:
                    results.append(_render_formula_worker(task))
                except Exception:
                    results.append((None, False, False))
    return results


def _run_render(formula_map: dict, do_graph: bool, n_workers: int) -> None:
    """Submit render tasks for formulas not yet on disk."""
    pending = [
        (vid, latex)
        for vid, latex in formula_map.items()
        if not (RENDERED_DIR / f"{vid}_typeset.png").exists()
    ]
    already = len(formula_map) - len(pending)
    print(f"  {already:,} already rendered, {len(pending):,} remaining")

    if not pending:
        print("Nothing to render.")
        return

    print(f"\nRendering with {n_workers} workers (graph={'yes' if do_graph else 'no'})...")

    tasks = [(vid, latex, str(RENDERED_DIR), do_graph) for vid, latex in pending]
    typeset_ok = graph_ok = typeset_fail = graph_fail = 0

    for _, t_ok, g_ok in _run_parallel_or_serial(tasks, "Rendering", n_workers):
        typeset_ok   += int(t_ok)
        typeset_fail += int(not t_ok)
        graph_ok     += int(g_ok)
        graph_fail   += int(not g_ok)

    print(
        f"\nRendering complete."
        f"\n  Typeset : {typeset_ok:,} ok  /  {typeset_fail:,} skipped or failed"
        f"\n  Graph   : {graph_ok:,} ok  /  {graph_fail:,} skipped or failed"
        f"\n  Output  : {RENDERED_DIR}"
    )


def _run_regraph(formula_map: dict, n_workers: int) -> None:
    """Retry graph rendering for all formulas that have a typeset but no graph."""
    pending = [
        (vid, latex)
        for vid, latex in formula_map.items()
        if (RENDERED_DIR / f"{vid}_typeset.png").exists()
        and not (RENDERED_DIR / f"{vid}_graph.png").exists()
        and _has_graph_potential(latex)
    ]
    print(f"  {len(pending):,} formulas with graph potential, no graph yet")

    if not pending:
        print("Nothing to regraph.")
        return

    print(f"\nRegraphing with {n_workers} workers...")

    tasks = [(vid, latex, str(RENDERED_DIR), True) for vid, latex in pending]
    graph_ok = graph_fail = 0

    for _, _, g_ok in _run_parallel_or_serial(tasks, "Regraphing", n_workers):
        graph_ok   += int(g_ok)
        graph_fail += int(not g_ok)

    print(
        f"\nRegraph complete."
        f"\n  Graph : {graph_ok:,} ok  /  {graph_fail:,} skipped or failed"
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main(
    render_all: bool = False,
    n_workers: int = None,
    prune: bool = False,
    dry_run: bool = False,
    regraph: bool = False,
) -> None:
    RENDERED_DIR.mkdir(parents=True, exist_ok=True)
    workers = n_workers or max(1, os.cpu_count() - 1)

    # ── Prune mode ────────────────────────────────────────────────────────────
    if prune:
        prune_bad_renders(dry_run=dry_run, n_workers=workers)
        return

    # ── Collect formula map (needed for render + regraph) ─────────────────────
    if render_all:
        print("Mode: full corpus (all formulas)")
        priority_ids = None
    else:
        print("Mode: priority posts (qrel-judged + image posts)")
        priority_ids = load_priority_post_ids()
        print(f"  {len(priority_ids):,} priority post IDs loaded")

    print("\nCollecting formula IDs from index...")
    formula_map = collect_formula_ids(priority_ids)
    print(f"  {len(formula_map):,} unique formulas")

    # ── Regraph mode ──────────────────────────────────────────────────────────
    if regraph:
        _run_regraph(formula_map, workers)
        return

    # ── Normal render mode ────────────────────────────────────────────────────
    _run_render(formula_map, do_graph=True, n_workers=workers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render formula images from the formula index")
    parser.add_argument(
        "--all",
        action="store_true",
        dest="render_all",
        help="Render all formulas in the corpus (default: priority posts only)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel worker processes (default: cpu_count - 1)",
    )
    parser.add_argument(
        "--prune",
        action="store_true",
        help="Audit and delete renders that fail quality checks",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="With --prune: show what would be deleted without deleting",
    )
    parser.add_argument(
        "--regraph",
        action="store_true",
        help="Retry graph rendering for all valid typeset formulas missing a graph",
    )
    args = parser.parse_args()
    main(
        render_all=args.render_all,
        n_workers=args.workers,
        prune=args.prune,
        dry_run=args.dry_run,
        regraph=args.regraph,
    )

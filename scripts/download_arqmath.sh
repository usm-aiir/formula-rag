#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# download_arqmath.sh
#
# Downloads the ARQMath dataset (v1.3) from the official RIT backup server:
#   https://www.cs.rit.edu/~dprl/ARQMath-backup/
#
# ARQMath Task Selection
# ──────────────────────
# This project uses all three ARQMath tasks:
#
#   Task 1 — Answer Retrieval (primary retrieval benchmark):
#     Given a question, retrieve relevant answer posts. qrels used for both
#     retrieval model training (ARQMath-1/2) and evaluation (ARQMath-3).
#
#   Task 2 — Formula Retrieval (formula encoder pre-training + image extension):
#     Given a formula query, retrieve similar formulas. qrels used to pre-train
#     the formula encoder (Stage 1 training) before multimodal alignment.
#     Also the natural task for validating that our image extension benefits
#     formula retrieval — the image extension is corpus-level and applies to
#     all tasks, so excluding Task 2 from evaluation would be incomplete.
#
#   Task 3 — Open Domain QA (generation benchmark):
#     Given a question, return a single generated answer. Used as the end-to-end
#     RAG pipeline evaluation.
#
# What gets downloaded (≈2.8 GB compressed, ≈7 GB unpacked):
#
#   data/raw/arqmath/
#     collection/
#       Posts.V1.3.zip            (895 MB) — main MSE corpus (2010–2018)
#       PostLinks.V1.3.xml         (29 MB) — related/duplicate post links
#       Tags.V1.3.xml             (169 KB) — tag vocabulary
#       README_DATA.md
#     formulas/
#       latex_representation_v3.zip  (406 MB) — LaTeX formula index
#       opt_representation_v3.zip    (587 MB) — Operator Trees (formula encoder training)
#       slt_representation_v3.zip    (835 MB) — Symbol Layout Trees (formula encoder training)
#       README_formulas_V3.0.md
#     topics/
#       task1/
#         arqmath1/Topics_V2.0.xml
#         arqmath2/Topics_Task1_2021_V1.1.xml
#         arqmath3/Topics_Task1_2022_V0.1.xml
#       task2/
#         arqmath1/Topics_V1.1.xml
#         arqmath2/Topics_Task2_2021_V1.1.xml
#         arqmath3/Topics_Task2_2022_V0.1.xml
#     qrels/
#       task1/
#         arqmath1/qrel_task1_2020_official
#         arqmath2/qrel_task1_2021_official.tsv
#         arqmath2/qrel_task1_2021_all.tsv
#         arqmath3/qrel_task1_2022_official.tsv
#         arqmath3/qrel_task1_2022_all.tsv
#       task2/
#         arqmath1/qrel_task2_2020_official.tsv
#         arqmath1/qrel_task2_2020_all.tsv
#         arqmath2/qrel_task2_2021_official.tsv
#         arqmath2/qrel_task2_2021_all.tsv
#         arqmath3/qrel_task2_2022_official.tsv
#         arqmath3/qrel_task2_2022_all.tsv
#     eval_scripts/
#       arqmath3/task1_get_results.py
#       arqmath3/arqmath_to_prime_task1.py
#       arqmath3/task2_get_results.py
#       arqmath3/de_duplicate_2022.py
#
# Usage:
#   bash scripts/download_arqmath.sh [--no-opt] [--no-slt]
#
#   --no-opt   Skip the Operator Tree formula index (saves 587 MB)
#   --no-slt   Skip the Symbol Layout Tree formula index (saves 835 MB)
#
# The script is idempotent: files that already exist are not re-downloaded.
# -----------------------------------------------------------------------------

set -euo pipefail

# ─── Parse flags ─────────────────────────────────────────────────────────────
DOWNLOAD_OPT=true
DOWNLOAD_SLT=true

for arg in "$@"; do
  case "$arg" in
    --no-opt) DOWNLOAD_OPT=false ;;
    --no-slt) DOWNLOAD_SLT=false ;;
    *)
      echo "Unknown argument: $arg"
      echo "Usage: $0 [--no-opt] [--no-slt]"
      exit 1
      ;;
  esac
done

# ─── Paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$REPO_ROOT/data/raw/arqmath"

COLLECTION_DIR="$DATA_DIR/collection"
FORMULAS_DIR="$DATA_DIR/formulas"
TOPICS_DIR="$DATA_DIR/topics"
QRELS_DIR="$DATA_DIR/qrels"
EVAL_DIR="$DATA_DIR/eval_scripts"

# ─── Base URLs ───────────────────────────────────────────────────────────────
BASE="https://www.cs.rit.edu/~dprl/ARQMath-backup"
TOPICS1_BASE="$BASE/Topics/Task1%263%20Topics"
TOPICS2_BASE="$BASE/Topics/Task2%20Topics"
EVAL_BASE="$BASE/Evaluation/%20Scripts%20%26%20Qrels"

# ─── Helpers ─────────────────────────────────────────────────────────────────
download() {
  local url="$1"
  local dest="$2"
  if [[ -f "$dest" ]]; then
    echo "  [skip] $(basename "$dest") already exists"
  else
    echo "  [download] $(basename "$dest")"
    curl --fail --silent --show-error --location \
         --retry 3 --retry-delay 5 \
         --output "$dest" \
         "$url"
  fi
}

banner() {
  echo ""
  echo "════════════════════════════════════════"
  echo "  $1"
  echo "════════════════════════════════════════"
}

# ─── Create directories ───────────────────────────────────────────────────────
mkdir -p \
  "$COLLECTION_DIR" \
  "$FORMULAS_DIR" \
  "$TOPICS_DIR/task1/arqmath1" \
  "$TOPICS_DIR/task1/arqmath2" \
  "$TOPICS_DIR/task1/arqmath3" \
  "$TOPICS_DIR/task2/arqmath1" \
  "$TOPICS_DIR/task2/arqmath2" \
  "$TOPICS_DIR/task2/arqmath3" \
  "$QRELS_DIR/task1/arqmath1" \
  "$QRELS_DIR/task1/arqmath2" \
  "$QRELS_DIR/task1/arqmath3" \
  "$QRELS_DIR/task2/arqmath1" \
  "$QRELS_DIR/task2/arqmath2" \
  "$QRELS_DIR/task2/arqmath3" \
  "$EVAL_DIR/arqmath3"

# ─── Collection ──────────────────────────────────────────────────────────────
banner "Collection (corpus)"
download \
  "$BASE/Collection/README_DATA.md" \
  "$COLLECTION_DIR/README_DATA.md"

download \
  "$BASE/Collection/Tags.V1.3.xml" \
  "$COLLECTION_DIR/Tags.V1.3.xml"

download \
  "$BASE/Collection/PostLinks.V1.3.xml" \
  "$COLLECTION_DIR/PostLinks.V1.3.xml"

echo ""
echo "  [note] Posts.V1.3.zip is 895 MB. Starting download..."
download \
  "$BASE/Collection/Posts.V1.3.zip" \
  "$COLLECTION_DIR/Posts.V1.3.zip"

# ─── Formulas ─────────────────────────────────────────────────────────────────
banner "Formula indexes"
download \
  "$BASE/Formulas/README_formulas_V3.0.md" \
  "$FORMULAS_DIR/README_formulas_V3.0.md"

echo ""
echo "  [note] latex_representation_v3.zip is 406 MB. Starting download..."
download \
  "$BASE/Formulas/latex_representation_v3.zip" \
  "$FORMULAS_DIR/latex_representation_v3.zip"

if [[ "$DOWNLOAD_OPT" == "true" ]]; then
  echo ""
  echo "  [note] opt_representation_v3.zip is 587 MB. Starting download..."
  download \
    "$BASE/Formulas/opt_representation_v3.zip" \
    "$FORMULAS_DIR/opt_representation_v3.zip"
else
  echo "  [skip] opt_representation_v3.zip (--no-opt)"
fi

if [[ "$DOWNLOAD_SLT" == "true" ]]; then
  echo ""
  echo "  [note] slt_representation_v3.zip is 835 MB. Starting download..."
  download \
    "$BASE/Formulas/slt_representation_v3.zip" \
    "$FORMULAS_DIR/slt_representation_v3.zip"
else
  echo "  [skip] slt_representation_v3.zip (--no-slt)"
fi

# ─── Task 1 Topics ───────────────────────────────────────────────────────────
banner "Task 1 topics (answer retrieval queries)"

download \
  "$TOPICS1_BASE/ARQMath-1/Topics/Topics_V2.0.xml" \
  "$TOPICS_DIR/task1/arqmath1/Topics_V2.0.xml"

download \
  "$TOPICS1_BASE/ARQMath-2/Topics/Topics_Task1_2021_V1.1.xml" \
  "$TOPICS_DIR/task1/arqmath2/Topics_Task1_2021_V1.1.xml"

download \
  "$TOPICS1_BASE/ARQMath-3/Topics/Topics_Task1_2022_V0.1.xml" \
  "$TOPICS_DIR/task1/arqmath3/Topics_Task1_2022_V0.1.xml"

# ─── Task 2 Topics ───────────────────────────────────────────────────────────
banner "Task 2 topics (formula retrieval queries)"

download \
  "$TOPICS2_BASE/ARQMath-1/Topics/Topics_V1.1.xml" \
  "$TOPICS_DIR/task2/arqmath1/Topics_V1.1.xml"

download \
  "$TOPICS2_BASE/ARQMath-2/Topics/Topics_Task2_2021_V1.1.xml" \
  "$TOPICS_DIR/task2/arqmath2/Topics_Task2_2021_V1.1.xml"

download \
  "$TOPICS2_BASE/ARQMath-3/Topics/Topics_Task2_2022_V0.1.xml" \
  "$TOPICS_DIR/task2/arqmath3/Topics_Task2_2022_V0.1.xml"

# ─── Task 1 Qrels ────────────────────────────────────────────────────────────
banner "Task 1 qrels (answer retrieval relevance judgments)"

download \
  "$EVAL_BASE/ARQMath-1/Task%201%20/qrel_official_task1" \
  "$QRELS_DIR/task1/arqmath1/qrel_task1_2020_official"

download \
  "$EVAL_BASE/ARQMath-2/Task1/Qrel%20Files/qrel_task1_2021_test.tsv" \
  "$QRELS_DIR/task1/arqmath2/qrel_task1_2021_official.tsv"

download \
  "$EVAL_BASE/ARQMath-2/Task1/Qrel%20Files/qrel_task1_2021_all.tsv" \
  "$QRELS_DIR/task1/arqmath2/qrel_task1_2021_all.tsv"

download \
  "$EVAL_BASE/ARQMath-3/Task%201/Qrel%20Files/qrel_task1_2022_official.tsv" \
  "$QRELS_DIR/task1/arqmath3/qrel_task1_2022_official.tsv"

download \
  "$EVAL_BASE/ARQMath-3/Task%201/Qrel%20Files/qrel_task1_2022_all.tsv" \
  "$QRELS_DIR/task1/arqmath3/qrel_task1_2022_all.tsv"

# ─── Task 2 Qrels ────────────────────────────────────────────────────────────
banner "Task 2 qrels (formula retrieval relevance judgments)"

download \
  "$EVAL_BASE/ARQMath-1/Task2/qrel_task2_official_2020_visual_id.tsv" \
  "$QRELS_DIR/task2/arqmath1/qrel_task2_2020_official.tsv"

download \
  "$EVAL_BASE/ARQMath-1/Task2/qrel_task2_2020_visual_id.tsv" \
  "$QRELS_DIR/task2/arqmath1/qrel_task2_2020_all.tsv"

download \
  "$EVAL_BASE/ARQMath-2/Task2/Qrel%20Files/qrel_task2_2021_test_official_evaluation.tsv" \
  "$QRELS_DIR/task2/arqmath2/qrel_task2_2021_official.tsv"

download \
  "$EVAL_BASE/ARQMath-2/Task2/Qrel%20Files/qrel_task2_2021_all.tsv" \
  "$QRELS_DIR/task2/arqmath2/qrel_task2_2021_all.tsv"

download \
  "$EVAL_BASE/ARQMath-3/Task2/Qrel%20Files/qrel_task2_2022_official.tsv" \
  "$QRELS_DIR/task2/arqmath3/qrel_task2_2022_official.tsv"

download \
  "$EVAL_BASE/ARQMath-3/Task2/Qrel%20Files/qrel_task2_2022_all.tsv" \
  "$QRELS_DIR/task2/arqmath3/qrel_task2_2022_all.tsv"

# ─── Eval scripts ─────────────────────────────────────────────────────────────
banner "Evaluation scripts"

download \
  "$EVAL_BASE/ARQMath-3/Task%201/task1_get_results.py" \
  "$EVAL_DIR/arqmath3/task1_get_results.py"

download \
  "$EVAL_BASE/ARQMath-3/Task%201/arqmath_to_prime_task1.py" \
  "$EVAL_DIR/arqmath3/arqmath_to_prime_task1.py"

download \
  "$EVAL_BASE/ARQMath-3/Task2/task2_get_results.py" \
  "$EVAL_DIR/arqmath3/task2_get_results.py"

download \
  "$EVAL_BASE/ARQMath-3/Task2/de_duplicate_2022.py" \
  "$EVAL_DIR/arqmath3/de_duplicate_2022.py"

# ─── Done ─────────────────────────────────────────────────────────────────────
banner "Done"
echo ""
echo "Downloaded to: $DATA_DIR"
echo ""
echo "Next steps:"
echo "  1. Unzip the corpus:"
echo "     cd $COLLECTION_DIR && unzip Posts.V1.3.zip"
echo ""
echo "  2. Unzip the formula index(es):"
echo "     cd $FORMULAS_DIR && unzip latex_representation_v3.zip"
if [[ "$DOWNLOAD_OPT" == "true" ]]; then
  echo "     cd $FORMULAS_DIR && unzip opt_representation_v3.zip"
fi
if [[ "$DOWNLOAD_SLT" == "true" ]]; then
  echo "     cd $FORMULAS_DIR && unzip slt_representation_v3.zip"
fi
echo ""
echo "  3. See docs/DATA_ACQUISITION.md for what to do next."
echo ""

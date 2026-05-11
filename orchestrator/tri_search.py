from __future__ import annotations

import argparse
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

import torch
from dotenv import load_dotenv
from huggingface_hub import login
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from clip_handler import fetch_clip_results
from gnn_handler import fetch_gnn_results
from utils.scrape_url import scrape_post_url

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

logger = logging.getLogger(__name__)

LLAMA_MODEL_NAME = os.getenv("LLAMA_MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
MAX_NEW_TOKENS = int(os.getenv("RAG_MAX_TOKENS", 512))

_SERVICE_TEXT = Path(__file__).resolve().parent.parent / "service-text"
if str(_SERVICE_TEXT) not in sys.path:
    sys.path.insert(0, str(_SERVICE_TEXT))

_model = None
_tokenizer = None
_pipeline = None
_text_handler = None


def load_model() -> None:
    """Load the Llama model from HuggingFace, attaching a LoRA adapter if one exists."""
    global _model, _tokenizer, _pipeline

    if HF_TOKEN:
        login(token=HF_TOKEN)

    torch.multiprocessing.set_start_method("spawn", force=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading model %s on %s", LLAMA_MODEL_NAME, device)

    _tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL_NAME)
    _tokenizer.pad_token = _tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        LLAMA_MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )

    adapter_path = os.path.join(
        os.path.dirname(__file__), "models", "mathmex-llama-dpo-adapter"
    )

    if os.path.exists(adapter_path):
        try:
            logger.info("Found LoRA adapter at %s. Merging weights...", adapter_path)
            _model = PeftModel.from_pretrained(base_model, adapter_path)
            logger.info("Adapter merged successfully.")
        except Exception as e:
            logger.warning("Failed to load adapter, using base model. Error: %s", e)
            _model = base_model
    else:
        logger.info("No LoRA adapter found. Using base model.")
        _model = base_model

    _pipeline = pipeline(
        "text-generation",
        model=_model,
        tokenizer=_tokenizer,
    )
    logger.info("Model loaded successfully.")


def _extract_final_answer(text: str) -> str:
    """
    Pull just the final answer out of model output that may contain step-by-step work.
    Priority order:
      1. LaTeX \\boxed{...} — the conventional "final answer" marker
      2. Last line that matches common answer-preamble patterns
      3. Last non-empty line as a fallback
    """
    import re

    # 1. \boxed{...} — may be nested, so match balanced braces manually
    boxed = re.findall(r"\\boxed\{([^{}]*)\}", text)
    if boxed:
        return f"${boxed[-1]}$"

    lines = [l.strip() for l in text.splitlines() if l.strip()]

    # 2. Line that starts with a known answer-preamble (case-insensitive)
    preamble = re.compile(
        r"^(?:the\s+(?:final\s+)?answer\s+is[:\s]*|answer[:\s]+|result[:\s]+|therefore[,\s]+|\$\$?)",
        re.IGNORECASE,
    )
    for line in reversed(lines):
        if preamble.match(line):
            # Strip the preamble words and return just the value
            return preamble.sub("", line).strip()

    # 3. Last non-empty line
    return lines[-1] if lines else text.strip()


def prompt_model(prompt: str) -> str:
    """Generate a response from the loaded Llama model."""
    global _pipeline

    if _pipeline is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")

    outputs = _pipeline(
        prompt,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        temperature=None,
        top_p=None,
    )
    generated: str = outputs[0].get("generated_text") or outputs[0].get("text") or ""

    if generated.startswith(prompt):
        generated = generated[len(prompt) :]

    return generated.strip()


def build_prompt(
    query: str,
    text_context: str,
    formula_context: str,
    image_context: str,
) -> str:
    """Assemble the structured LLM prompt from retrieved context blocks."""
    text_block    = text_context    if text_context    else "(no text documents retrieved)"
    formula_block = f"\nFormula-matched posts:\n{formula_context}\n" if formula_context else ""
    clipvb_block  = f"\nLLM2CLIP image posts:\n{image_context}\n" if image_context  else ""

    return (
        "<|system|>\n"
        "You are a mathematical answer engine. Respond with the final answer only "
        "\u2014 a single expression or number, nothing else. "
        "No steps, no explanation, no preamble.\n"
        "<|user|>\n"
        f"Documents:\n{text_block}\n"
        f"{formula_block}"
        f"{clipvb_block}"
        f"Question: {query}\n"
        "<|assistant|>\n"
    )




def _formula_structure_score(input_formula: str, result_formula: str) -> float:
    """
    Calculate a structural similarity score between input and result formulas.
    Considers formula length similarity and symbol overlap.
    Returns a score between 0 and 1.
    """
    import re

    # Extract symbols (alphanumeric and common math operators)
    def extract_symbols(formula: str) -> set:
        return set(re.findall(r'[a-zA-Z0-9\+\-\*\/\=\(\)\[\]\{\}]', formula))

    input_len = len(input_formula)
    result_len = len(result_formula)
    if input_len == 0 or result_len == 0:
        length_sim = 0.0
    else:
        length_ratio = min(input_len, result_len) / max(input_len, result_len)
        length_sim = length_ratio

    input_symbols = extract_symbols(input_formula)
    result_symbols = extract_symbols(result_formula)

    if not input_symbols and not result_symbols:
        symbol_sim = 1.0
    elif not input_symbols or not result_symbols:
        symbol_sim = 0.0
    else:
        intersection = len(input_symbols & result_symbols)
        union = len(input_symbols | result_symbols)
        symbol_sim = intersection / union if union > 0 else 0.0

    return (length_sim + symbol_sim) / 2.0


def clipvb_sources(query: str, k: int = 5) -> str:
    """
    Find the k most relevant images using LLM2CLIP and return the combined
    scraped post text for use as RAG context.
    """
    hits = fetch_clip_results(query, k=k)
    if not hits:
        logger.warning("Image search returned no results.")
        return ""
    parts = [
        f"[LLM2CLIP image source: {h.get('title') or h.get('url', '')}]\n{h['scraped_text']}"
        for h in hits
        if h.get("scraped_text")
    ]
    return "\n\n".join(parts)



def formula_sources(query: str, top_k: int = 5) -> str:
    """
    Extract LaTeX formulas from the query, retrieve similar formulas via the
    GNN encoder, and return the combined scraped post text for use as RAG context.
    """
    hits = fetch_gnn_results(query, k=top_k)
    if not hits:
        return ""
    parts = [
        f"[Formula source: {h['url']}]\n{h['scraped_text']}"
        for h in hits
        if h.get("scraped_text")
    ]
    return "\n\n".join(parts)

    return "\n\n".join(parts)

def text_sources(query: str, top_k: int = 5) -> str:
    """Retrieve top-k text passages from OpenSearch using TextHandler directly."""
    global _text_handler
    if _text_handler is None:
        from text_handler import TextHandler
        _text_handler = TextHandler()
    return _text_handler.retrieve_relevant_text(query, top_k=top_k)

def re_rank_sources(query: str, text_sources: List[dict], formula_sources: List[dict], image_sources: List[dict]) -> List[dict]:
    """
    Re-rank retrieved sources by relevance to the query using the LLM.
    
    if just text sources are available, return them as is
    if text and formulas or images are available, use min max to normalize the scores, 
    then combine them with a weighted average (e.g., 0.2 text, 0.3 formulas, 0.5 images)
    to get an overall relevance score for each source. favor images, then formulas, then text, 
    since images are likely to be most relevant if needed,
    but formulas can be a strong signal too.
    """
    # If only text sources are available, return them as is
    if not formula_sources and not image_sources:
        return text_sources

    # Normalize scores for each modality - uses min max
    def normalize_scores(sources: List[dict], score_key: str) -> List[dict]:
        if not sources:
            return sources
        scores = [s.get(score_key, 0.0) for s in sources]
        min_score, max_score = min(scores), max(scores)
        for s in sources:
            raw_score = s.get(score_key, 0.0)
            s[f"{score_key}_norm"] = (raw_score - min_score) / (max_score - min_score + 1e-8)
        return sources

    text_sources = normalize_scores(text_sources, "text_score")
    formula_sources = normalize_scores(formula_sources, "formula_score")
    image_sources = normalize_scores(image_sources, "image_score")

    # now calculate the relevence weight. .5 if two sources, .33 if three sources
    relevance_weight = 1.0
    if text_sources and (formula_sources or image_sources):
        relevance_weight = 0.5
    if text_sources and formula_sources and image_sources:
        relevance_weight = 0.33

    # Combine all sources into a single list with weighted relevance scores
    combined_sources = []
    for source in text_sources:
        combined_sources.append({**source, "relevance": relevance_weight * source.get("text_score_norm", 0.0)})
    for source in formula_sources:
        combined_sources.append({**source, "relevance": relevance_weight * source.get("formula_score_norm", 0.0)})
    for source in image_sources:
        combined_sources.append({**source, "relevance": relevance_weight * source.get("image_score_norm", 0.0)})

    # Sort by combined relevance score
    combined_sources.sort(key=lambda x: x.get("relevance", 0.0), reverse=True)
    
    # return the top ten
    return combined_sources[:10]

def rag_query(
    query: str, top_k_text: int = 10, top_k_formulas: int = 10, top_k_images: int = 10
) -> str:
    """
    Full RAG pipeline: retrieve text + formula + image sources (both CLIP and LLM2CLIP),
    build a grounded prompt, then generate a response from the LLM.
    """
    # Retrieve from all modalities in parallel to minimise wall-clock time.
    with ThreadPoolExecutor(max_workers=3) as executor:
        fut_text     = executor.submit(text_sources,   query, top_k_text)
        fut_formulas = executor.submit(formula_sources, query, top_k_formulas)
        fut_clipvb   = executor.submit(clipvb_sources, query, top_k_images)

    prompt = build_prompt(
        query,
        text_context=fut_text.result(),
        formula_context=fut_formulas.result(),
        image_context=fut_clipvb.result(),
    )
    return prompt_model(prompt)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Tri-search RAG query.")
    parser.add_argument("query", nargs="?", help="Question to ask.")
    args = parser.parse_args()

    load_model()

    if args.query:
        print(rag_query(args.query))
    else:
        print("Model loaded. Enter prompts (Ctrl-C to exit):")
        while True:
            try:
                user_input = input("> ")
                if user_input.strip():
                    print(rag_query(user_input))
            except (KeyboardInterrupt, EOFError):
                break

from __future__ import annotations

import argparse
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

import requests
import torch
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from huggingface_hub import login
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

logger = logging.getLogger(__name__)

LLAMA_MODEL_NAME = os.getenv("LLAMA_MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
MAX_NEW_TOKENS = int(os.getenv("RAG_MAX_TOKENS", 512))

_model = None
_tokenizer = None
_pipeline = None


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

    return _extract_final_answer(generated.strip())


def scrape_post_url(url: str, max_chars: int = 1500) -> str:
    """Scrape question and top-2 answers from any Stack Exchange post URL."""
    try:
        headers = {"User-Agent": "TriSearchRAG/0.1"}
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        parts: List[str] = []
        question_body = soup.select_one(".question .s-prose")
        if question_body:
            parts.append("Question:\n" + question_body.get_text(separator=" ", strip=True))
        for answer in soup.select(".answer .s-prose")[:2]:
            parts.append("Answer:\n" + answer.get_text(separator=" ", strip=True))

        text = "\n\n".join(parts).strip()
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
        return text
    except Exception as err:
        logger.warning("Could not scrape %s: %s", url, err)
        return ""


def image_sources(query: str, k: int = 5) -> str:
    """
    Find the k most relevant images for the query, scrape the source post
    for each, and return the combined text.
    """
    from image_handler import search

    results = search(query, k=k)
    parts: List[str] = []
    seen_urls: set = set()
    for r in results:
        url = r.get("url", "")
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        post_text = scrape_post_url(url)
        if post_text:
            title = r.get("title") or url
            parts.append(f"[Image source: {title}]\n{post_text}")
    return "\n\n".join(parts)


def formula_sources(query: str, top_k: int = 5) -> str:
    """
    Extract LaTeX formulas from the query, retrieve similar formulas via TangentCFT,
    scrape the originating MSE posts, and return the combined text.
    """
    from formula_handler import FormulaHandler
    from formula_utils import extract_formulas

    formulas = extract_formulas(query)
    if not formulas:
        logger.info("No formulas detected in query; skipping formula retrieval.")
        return ""

    logger.info("Retrieving formula sources for: %s", formulas)
    handler = FormulaHandler(formulas)
    results = handler.retrieve_similar_formulas(top_k=top_k)

    parts: List[str] = []
    seen_threads: set = set()
    for hit in results:
        thread_id = hit.get("thread_id")
        if not thread_id or thread_id in seen_threads:
            continue
        seen_threads.add(thread_id)

        post_text = scrape_post_url(f"https://math.stackexchange.com/q/{thread_id}")
        if post_text:
            header = hit.get("original_question") or f"MSE post #{thread_id}"
            parts.append(
                f"[Formula Match: {hit.get('returned_formula', '')}]\n{header}\n{post_text}"
            )

    return "\n\n".join(parts)


def text_sources(query: str, top_k: int = 5) -> str:
    """Return the top-k relevant text passages from OpenSearch for the given query."""
    from text_handler import TextHandler

    handler = TextHandler()
    return handler.retrieve_relevant_text(query, top_k=top_k)


def rag_query(
    query: str, top_k_text: int = 5, top_k_formulas: int = 5, top_k_images: int = 3
) -> str:
    """
    Full RAG pipeline: retrieve text + formula + image sources, build a grounded
    prompt, then generate a response from the LLM.
    """
    # Run the three retrieval tasks in parallel to save time
    with ThreadPoolExecutor(max_workers=3) as executor:
        fut_text = executor.submit(text_sources, query, top_k_text)
        fut_formulas = executor.submit(formula_sources, query, top_k_formulas)
        fut_images = executor.submit(image_sources, query, top_k_images)
        retrieved_text = fut_text.result()
        retrieved_formulas = fut_formulas.result()
        image_text = fut_images.result()

    text_block     = retrieved_text   if retrieved_text   else "(no text documents retrieved)"
    formula_block  = f"\nFormula-matched posts:\n{retrieved_formulas}\n" if retrieved_formulas else ""
    image_block    = f"\nImage-sourced posts:\n{image_text}\n"           if image_text         else ""

    prompt = f"""<|system|>
You are a mathematical answer engine. Respond with the final answer only — a single expression or number, nothing else. No steps, no explanation, no preamble.
<|user|>
Documents:
{text_block}
{formula_block}{image_block}
Question: {query}
<|assistant|>
"""
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

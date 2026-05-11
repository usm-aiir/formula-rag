import sys
import logging
from typing import List

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


def scrape_post_url(url: str) -> str:
    """Scrape question and top-2 answers from any Stack Exchange post URL.

    Returns the full text of the question body and up to two answers,
    separated by blank lines, with no character truncation.
    """
    try:
        headers = {"User-Agent": "TriSearchRAG/0.1"}
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        parts: List[str] = []
        question_body = soup.select_one(".question .s-prose")
        if question_body:
            parts.append(
                "Question:\n" + question_body.get_text(separator=" ", strip=True)
            )
        for answer in soup.select(".answer .s-prose")[:2]:
            parts.append("Answer:\n" + answer.get_text(separator=" ", strip=True))

        return "\n\n".join(parts).strip()
    except Exception as err:
        logger.warning("Could not scrape %s: %s", url, err)
        return ""


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    TEST_URL = "https://math.stackexchange.com/questions/9508/how-to-solve-a-quadratic-equation"

    parser = argparse.ArgumentParser(description="Test scrape_post_url")
    parser.add_argument(
        "url",
        nargs="?",
        default=TEST_URL,
        help="Stack Exchange post URL to scrape (default: a quadratic equation post)",
    )
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"URL: {args.url}")
    print(f"{'='*60}\n")

    result = scrape_post_url(args.url)
    if not result:
        print("No content scraped. Check the URL or your internet connection.")
        sys.exit(1)

    blocks = result.split("\n\n")
    for i, block in enumerate(blocks, start=1):
        print(f"--- Block {i} ---")
        print(block)
        print()

    print(f"Total characters scraped: {len(result):,}")

# Currently is dealing with auth errors, look into this

# Note: Enables code to be more flexible, postpones evaluation of type annotations
from __future__ import annotations

import logging
import os
import sys
from typing import List, Optional

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from opensearchpy import OpenSearch
from sentence_transformers import SentenceTransformer
from schemas.TextRetrievalResult import TextRetrievalResult

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.scrape_url import scrape_post_url

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
logger = logging.getLogger(__name__)

class TextHandler:

    def __init__(self) -> None:
        self.model_path = os.path.join(os.path.dirname(__file__), "arq1thru3-finetuned-all-mpnet-jul-27")
        self.indices: List[str] = [
            "mathmex_math-overflow",
            "mathmex_math-stack-exchange",
            "mathmex_mathematica",
            "mathmex_wikipedia",
            "mathmex_youtube",
        ]
        self._text_model: Optional[SentenceTransformer] = None
        self._opensearch_client: Optional[OpenSearch] = None

    def _load_model(self) -> SentenceTransformer:
        if self._text_model is None:
            self._text_model = SentenceTransformer(str(self.model_path))
        return self._text_model

    def _create_opensearch_client(self) -> OpenSearch:
        host = os.getenv("OPENSEARCH_HOST")
        port = int(os.getenv("OPENSEARCH_PORT", "9200"))
        user = os.getenv("OPENSEARCH_USER")
        password = os.getenv("OPENSEARCH_PASSWORD")
        if not host or not user or not password:
            raise ValueError(
                "OPENSEARCH_HOST, OPENSEARCH_USER, and OPENSEARCH_PASSWORD "
                "environment variables must be set"
            )

        return OpenSearch(
        hosts=[{
            "host": host,
            "port": port,
        }],
        http_auth=(
           user,
            password,
        ),
        use_ssl=True,
        verify_certs=False,
        ssl_show_warn=False,
    )

    def _get_client(self) -> OpenSearch:
        if self._opensearch_client is None:
            self._opensearch_client = self._create_opensearch_client()
        return self._opensearch_client

    def retrieve_relevant_text(
        self,
        query: str,
        top_k: int = 200,
    ) -> str:

        formatted_query = query.lower().strip()
        text_model = self._load_model()
        opensearch_client = self._get_client()

        try:
            query_text = BeautifulSoup(formatted_query, "lxml").text.replace("$", "")
            if not query_text.strip():
                logger.warning(
                    "Query text is empty after HTML stripping, using original"
                )
                query_text = query
            query_vector = text_model.encode(query_text).tolist()
        except Exception as error:
            logger.error("Failed to encode query text: %s", error)
            return ""

        indices_to_search = self.indices

        if not indices_to_search:
            logger.warning("No valid indices to search")
            return ""

        # todo: could this be an obj? I feel it would be clearn than this jumbled mess
        search_query = {
            "from": 0,
            "size": int(top_k),
            "_source": {"includes": ["title", "media_type", "body_text", "link"]},
            "query": {
                "bool": {
                    "must": [
                        {
                            "knn": {
                                "body_vector": {
                                    "vector": query_vector,
                                    "k": int(top_k),
                                }
                            }
                        }
                    ]
                }
            },
        }

        text_search_results: List[TextRetrievalResult] = []
        try:
            search_response = opensearch_client.search(
                index=indices_to_search, body=search_query
            )
            for position, hit in enumerate(
                search_response.get("hits", {}).get("hits", []), start=1
            ):
                source = hit.get("_source", {})
                link = str(source.get("link", ""))
                text_search_results.append(
                    TextRetrievalResult(
                        doc_id=link or str(hit.get("_id", f"doc_{position}")),
                        score=float(hit.get("_score", 0.0) or 0.0),
                        rank=position,
                        text=str(source.get("body_text", "")),
                    )
                )
        except Exception as error:
            logger.error("Text search failed: %s", error)

        logger.info("Text search complete: %s results found", len(text_search_results))
        formatted_sources = []
        for result in text_search_results:
            url = result.doc_id
            if url.startswith("http"):
                scraped = scrape_post_url(url)
                content = scraped if scraped else result.text
            else:
                content = result.text
            formatted_sources.append(f"[Source {result.rank}]\n{content}")
        return "\n\n".join(formatted_sources)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    handler = TextHandler()
    result = handler.retrieve_relevant_text(query="What is a Taylor series?", top_k=5)
    print(result)

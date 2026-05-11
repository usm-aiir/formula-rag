# Note: Enables code to be more flexible, postpones evaluation of type annotations
from __future__ import annotations

import logging
import os
from typing import List, Optional

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from opensearchpy import OpenSearch
from sentence_transformers import SentenceTransformer
from schemas.TextRetrievalResult import TextRetrievalResult

from utils.scrape_url import scrape_post_url

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
logger = logging.getLogger(__name__)

class TextHandler:

    def __init__(self) -> None:
        self.model_path = os.path.join(os.path.dirname(__file__), "arq1thru3-finetuned-all-mpnet-jul-27")
        # todo: should be a global import from orchestrator
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
            hosts=[{"host": host, "port": port}],
            http_auth=(user, password),
            use_ssl=True,
            verify_certs=False,
            ssl_show_warn=False,
        )

    def _get_client(self) -> OpenSearch:
        if self._opensearch_client is None:
            self._opensearch_client = self._create_opensearch_client()
        return self._opensearch_client

    def search(
        self,
        query: str,
        top_k: int = 200,
    ) -> List[TextRetrievalResult]:
        """Query OpenSearch and return raw ranked hits. Scraping is the caller's responsibility."""
        text_model = self._load_model()
        opensearch_client = self._get_client()

        try:
            cleaned_query = BeautifulSoup(query.lower().strip(), "lxml").text.replace("$", "")
            if not cleaned_query.strip():
                logger.warning("Query text is empty after HTML stripping, using original")
                cleaned_query = query
            query_vector = text_model.encode(cleaned_query)
        except Exception as error:
            logger.error("Failed to encode query text: %s", error)
            return []

        if not self.indices:
            logger.warning("No valid indices to search")
            return []

        search_query = {
            "from": 0,
            "size": top_k,
            "_source": {"includes": ["title", "media_type", "body_text", "link"]},
            "query": {
                "bool": {
                    "must": [{"knn": {"body_vector": {"vector": query_vector, "k": top_k}}}]
                }
            },
        }

        results: List[TextRetrievalResult] = []
        try:
            search_response = opensearch_client.search(
                index=self.indices, body=search_query
            )
            for position, hit in enumerate(
                search_response.get("hits", {}).get("hits", []), start=1
            ):
                source = hit.get("_source", {})
                link = str(source.get("link", ""))
                results.append(
                    TextRetrievalResult(
                        doc_id=link or str(hit.get("_id", f"doc_{position}")),
                        score=float(hit.get("_score", 0.0) or 0.0),
                        rank=position,
                        text=str(source.get("body_text", "")),
                    )
                )
        except Exception as error:
            logger.error("Text search failed: %s", error)

        logger.info("Text search complete: %d results found", len(results))
        return results

    def retrieve_relevant_text(
        self,
        query: str,
        top_k: int = 200,
    ) -> str:
        """This method uses the search method to get relevant documents, then scrapes the content to then be used in the rag"""
        results = self.search(query, top_k=top_k)
        formatted_sources = []
        for result in results:
            url = result.doc_id
            if url.startswith("http"):
                scraped = scrape_post_url(url)
                content = scraped if scraped else result.text
            else:
                content = result.text
            formatted_sources.append(f"[Source {result.rank}]\n{content}")
        return "\n\n".join(formatted_sources)


if __name__ == "__main__":
    # testing
    logging.basicConfig(level=logging.INFO)
    handler = TextHandler()
    result = handler.retrieve_relevant_text(query="What is a Taylor series?", top_k=5)
    print(result)

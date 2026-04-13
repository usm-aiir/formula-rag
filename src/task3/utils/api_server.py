"""
Formula Retrieval API Server

Exposes the Tri-RAG Formula Retrieval Engine via a REST API.
Loads models into VRAM on startup and listens for incoming POST requests.
"""

import sys
from pathlib import Path
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

# --- Pathing ---
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.task3.utils.formula_retriever import FormulaRetriever

app = FastAPI(title="Tri-RAG Formula Retrieval API", version="1.0")

# Define the global variable, but don't load the models yet
retriever = None

@app.on_event("startup")
def startup_event():
    """Initialize globally so it loads into VRAM only once on server boot."""
    global retriever
    print("Booting up FastAPI Server and ML Models...")
    try:
        retriever = FormulaRetriever()
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to initialize FormulaRetriever: {e}")
        retriever = None

# --- Pydantic Data Models for JSON Validation ---
class SearchRequest(BaseModel):
    query: str
    top_k: int = 10

class SearchResult(BaseModel):
    rank: int
    visual_id: str
    score: float

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]

# --- API Endpoints ---
@app.get("/health")
def health_check():
    """Simple endpoint to verify the server and ML models are active."""
    if retriever is None:
        raise HTTPException(status_code=500, detail="Retriever engine failed to load.")
    return {"status": "healthy", "message": "Formula RAG engine is online and ready."}

@app.post("/search", response_model=SearchResponse)
def search_formulas(req: SearchRequest):
    """
    Main inference endpoint. 
    Accepts a JSON payload with a LaTeX string and returns ranked Visual IDs.
    """
    if retriever is None:
        raise HTTPException(status_code=500, detail="Retriever engine offline.")
        
    try:
        # Run the search pipeline
        raw_results = retriever.search(req.query, final_top_k=req.top_k)
        
        # Format the output
        formatted_results = []
        for rank, (vid, score) in enumerate(raw_results, 1):
            formatted_results.append({
                "rank": rank,
                "visual_id": str(vid),
                "score": float(score)  # Ensure type compatibility for JSON
            })
            
        return {
            "query": req.query,
            "results": formatted_results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

if __name__ == "__main__":
    # Run the server on port 8567
    print("\nStarting Uvicorn server on http://0.0.0.0:8567 ...")
    uvicorn.run("api_server:app", host="0.0.0.0", port=8567, reload=False)
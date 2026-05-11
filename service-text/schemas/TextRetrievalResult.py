from pydantic import BaseModel
# legacy, should be swapped out for something else
class TextRetrievalResult(BaseModel):
    doc_id: str
    score: float
    rank: int
    text: str
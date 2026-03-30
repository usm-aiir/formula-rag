from pydantic import BaseModel


class TextRetrievalResult(BaseModel):
    doc_id: str
    score: float
    rank: int
    text: str
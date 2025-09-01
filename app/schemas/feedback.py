from pydantic import BaseModel


class FeedbackRequest(BaseModel):
    taxonomy: str  # symbol
    query: str
    reference: str
    tag: str
    is_correct: bool
    is_custom: bool
    rank: int


class FeedbackResponse(BaseModel):
    message: str
    saved: bool

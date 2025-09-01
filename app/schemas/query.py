from typing import List, Optional
from pydantic import BaseModel


class BuildRequest(BaseModel):
    taxonomy: str  # taxonomy symbol


class QueryRequest(BaseModel):
    query: str
    taxonomy: str
    k: int = 5
    rerank: bool = True


class QueryResult(BaseModel):
    tag: str
    datatype: str
    reference: str
    score: float
    rank: int


class QueryResponse(BaseModel):
    query: str
    taxonomy: str
    results: List[QueryResult]

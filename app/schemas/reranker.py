from pydantic import BaseModel
from typing import Optional


class RerankerRequest(BaseModel):
    name: Optional[str] = None
    version: Optional[str] = None
    path: Optional[str] = None
    normalize_method: Optional[str] = None

from pydantic import BaseModel
from typing import Optional


class EmbedderDTO(BaseModel):
    id: int
    name: str
    version: str
    path: str
    is_active: bool

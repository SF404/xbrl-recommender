from pydantic import BaseModel
from typing import Optional


class UpdateSettingsRequest(BaseModel):
    active_embedder_id: int
    active_reranker_id: int


class SettingsResponse(BaseModel):
    id: int
    active_embedder_id: int
    active_reranker_id: int
    updated_at: Optional[str]

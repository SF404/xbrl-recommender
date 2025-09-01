from typing import List, Dict, Any
import numpy as np
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from sqlalchemy.orm import Session
from ..core.config import get_settings
from ..core.errors import AppException, ErrorCode
from ..models.entities import Setting, Embedder, Reranker
from .initialize_db import initialize_db 

settings = get_settings()

class RerankerWrapper:
    def __init__(self, model_name: str, device: str = "cpu", normalize_method: str = "softmax"):
        self.model = CrossEncoder(model_name, device=device)
        self.normalize_method = normalize_method

    def rerank(self, query: str, docs: List[Document], top_k: int = 5):
        pairs = [(query, d.metadata["reference"]) for d in docs]
        raw = self.model.predict(pairs)

        if self.normalize_method == "softmax":
            exp_scores = np.exp(raw - np.max(raw))
            scores = exp_scores / exp_scores.sum()
        elif self.normalize_method == "sigmoid":
            scores = 1 / (1 + np.exp(-raw))
        elif self.normalize_method == "minmax":
            min_s, max_s = np.min(raw), np.max(raw)
            scores = (raw - min_s) / (max_s - min_s) if max_s != min_s else np.ones_like(raw) * 0.5
        else:
            scores = raw

        ranked = list(zip(docs, scores))
        return sorted(ranked, key=lambda x: x[1], reverse=True)[:top_k]


class ModelRegistry:
    """
    Holds live references to the active embedder and reranker.
    Loaded once at startup and on-demand via /reload_models.
    """
    def __init__(self):
        self.settings = get_settings()
        self.embedder = None
        self.reranker = None

    def ensure_loaded(self):
        if self.embedder is None or self.reranker is None:
            raise AppException(ErrorCode.MODEL_NOT_LOADED, "Active models are not loaded.", status_code=503)

    def load_from_db(self, db: Session):
        """
        Loads embedder and reranker models from the database settings.
        If no settings exist, initialize them.
        """
        setting = db.query(Setting).first()
        print(setting)

        # If no setting exists, initialize DB
        if not setting:
            print("[ModelRegistry] No settings found, initializing database.")
            initialize_db()  # Call initialize_db to populate embedder and reranker
            setting = db.query(Setting).first()  # Re-fetch after initialization

        # Ensure there's only one setting entry in the DB
        self._ensure_single_setting(db)

        embedder = db.query(Embedder).filter(Embedder.id == setting.active_embedder_id).first()
        rer = db.query(Reranker).filter(Reranker.id == setting.active_reranker_id).first()

        if not embedder or not rer:
            raise AppException(
                ErrorCode.SETTINGS_NOT_FOUND,
                "Active embedder or reranker not found for current settings.",
                status_code=500,
            )

        self.embedder = HuggingFaceEmbeddings(model_name=embedder.path, model_kwargs={"device": self.settings.DEVICE})
        self.reranker = RerankerWrapper(model_name=rer.path, device=self.settings.DEVICE, normalize_method=rer.normalize_method)
        print(f"Loaded embedder: {embedder.name} ({embedder.version}) from {embedder.path}")
        print(f"Loaded reranker: {rer.name} ({rer.version}) from {rer.path}")
        return {"embedder": embedder, "reranker": rer}

    def _ensure_single_setting(self, db: Session):
        """
        Ensures that only one setting record exists.
        Deletes any extra setting records.
        """
        settings = db.query(Setting).all()
        if len(settings) > 1:
            print("[ModelRegistry] More than one setting found. Deleting excess settings.")
            for extra_setting in settings[1:]:
                db.delete(extra_setting)
            db.commit()
            print("[ModelRegistry] Excess settings deleted. Only one active setting exists.")

# app/services/training.py
import os
from typing import Optional

from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, CrossEncoder, InputExample, losses
from sqlalchemy.orm import Session

from ..core.config import get_settings
from ..core.errors import AppException, ErrorCode
from ..models.entities import Embedder, Reranker, Setting
from .feedback_service import load_feedback_pairs

settings = get_settings()


# ---------- helpers ----------
def _parse_version(v: Optional[str]) -> int:
    """Parse a version string like 'v12' -> 12; return 0 if None/invalid."""
    if not v:
        return 0
    try:
        return int(v.lstrip("v").strip())
    except Exception:
        return 0


def _next_version(db: Session, Model) -> str:
    """Compute the next numeric version across all rows of a model table."""
    versions = [row[0] for row in db.query(Model.version).all()]
    max_n = 0
    for v in versions:
        n = _parse_version(v)
        if n > max_n:
            max_n = n
    return f"v{max_n + 1}"


def _activate_models(db: Session, embedder_id: Optional[int] = None, reranker_id: Optional[int] = None):
    """Update Setting to point to newly trained model(s)."""
    setting = db.query(Setting).first()
    if not setting:
        setting = Setting(active_embedder_id=embedder_id, active_reranker_id=reranker_id)
        db.add(setting)
    else:
        if embedder_id is not None:
            setting.active_embedder_id = embedder_id
        if reranker_id is not None:
            setting.active_reranker_id = reranker_id
    db.commit()


def _ensure_active_records(db: Session) -> Setting:
    setting = db.query(Setting).first()
    if not setting:
        raise AppException(
            ErrorCode.SETTINGS_NOT_FOUND,
            "No settings found in DB. Seed at least one embedder/reranker and a Setting row.",
            status_code=500,
        )
    return setting


# ---------- training functions ----------
def train_embedder(db: Session) -> str:
    """
    Fine-tune the SentenceTransformer starting from the **active embedder path** in DB.
    Saves to FINETUNE_DIR/embedder_<new_version>, inserts a new Embedder row, and activates it.
    """
    pairs = load_feedback_pairs(db)
    if not pairs:
        return "No feedback data found."

    # start from ACTIVE embedder
    setting = _ensure_active_records(db)
    active_emb: Embedder = db.query(Embedder).filter(Embedder.id == setting.active_embedder_id).first()
    if not active_emb or not active_emb.path:
        raise AppException(ErrorCode.NOT_FOUND, "Active embedder not found or path missing.", status_code=404)

    # training data (positives only for MNRL)
    train_examples = [InputExample(texts=[q, ref], label=1.0) for q, ref, lbl in pairs if lbl == 1]
    if not train_examples:
        return "No positive feedback pairs for embedder training."

    # load from active embedder PATH (local or hub id); place on configured device
    model = SentenceTransformer(active_emb.path, device=settings.DEVICE)

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=5,
        warmup_steps=10,
        show_progress_bar=True,
    )

    # save with a new versioned path
    new_version = _next_version(db, Embedder)
    save_path = os.path.join(settings.FINETUNE_DIR, f"embedder_{new_version}")
    os.makedirs(save_path, exist_ok=True)
    model.save(save_path)

    # record in DB & activate
    emb = Embedder(
        name=f"fine-tuned_embedder_{new_version}",
        version=new_version,
        path=save_path,
        is_active=True,
    )
    db.add(emb)
    db.commit()
    db.refresh(emb)

    _activate_models(db, embedder_id=emb.id)

    return f"Embedder fine-tuned from '{active_emb.path}' → '{save_path}' (version {new_version}, active id={emb.id})."


def train_reranker(db: Session) -> str:
    """
    Fine-tune the CrossEncoder starting from the **active reranker path** in DB.
    Saves to FINETUNE_DIR/reranker_<new_version>, inserts a new Reranker row, and activates it.
    """
    pairs = load_feedback_pairs(db)
    if not pairs:
        return "No feedback data found."

    # start from ACTIVE reranker
    setting = _ensure_active_records(db)
    active_rer: Reranker = db.query(Reranker).filter(Reranker.id == setting.active_reranker_id).first()
    if not active_rer or not active_rer.path:
        raise AppException(ErrorCode.NOT_FOUND, "Active reranker not found or path missing.", status_code=404)

    # training data: use label as float score (0/1 typical)
    train_examples = [InputExample(texts=[q, ref], label=float(lbl)) for q, ref, lbl in pairs]
    if not train_examples:
        return "No feedback pairs for reranker training."

    # load CrossEncoder from active path (local dir saved via CrossEncoder.save, or hub id)
    # NOTE: num_labels=1 ensures regression-style scoring appropriate for reranking
    model = CrossEncoder(active_rer.path, num_labels=1, device=settings.DEVICE)

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

    model.fit(
        train_dataloader=train_dataloader,
        epochs=5,
        warmup_steps=10,
        show_progress_bar=True,
    )

    # save with a new versioned path
    new_version = _next_version(db, Reranker)
    save_path = os.path.join(settings.FINETUNE_DIR, f"reranker_{new_version}")
    os.makedirs(save_path, exist_ok=True)
    model.save(save_path)  # IMPORTANT: saves classifier head too

    # record in DB & activate
    rer = Reranker(
        name=f"fine-tuned_reranker_{new_version}",
        version=new_version,
        path=save_path,
        normalize_method=active_rer.normalize_method or "softmax",
        is_active=True,
    )
    db.add(rer)
    db.commit()
    db.refresh(rer)

    _activate_models(db, reranker_id=rer.id)

    return f"Reranker fine-tuned from '{active_rer.path}' → '{save_path}' (version {new_version}, active id={rer.id})."

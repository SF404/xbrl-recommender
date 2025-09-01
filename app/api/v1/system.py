from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ...core.deps import get_db, get_registry
from ...core.errors import AppException, ErrorCode
from ...schemas.settings import UpdateSettingsRequest, SettingsResponse
from ...models.entities import Setting, Embedder, Reranker

router = APIRouter()


@router.get("/active_models")
def active_models(db: Session = Depends(get_db)):
    setting = db.query(Setting).first()
    if not setting:
        raise AppException(ErrorCode.SETTINGS_NOT_FOUND, "No settings found", status_code=404)
    emb = db.query(Embedder).filter(Embedder.id == setting.active_embedder_id).first()
    rer = db.query(Reranker).filter(Reranker.id == setting.active_reranker_id).first()
    return {
        "embedder": {"id": emb.id, "version": emb.version, "path": emb.path},
        "reranker": {"id": rer.id, "version": rer.version, "path": rer.path, "normalize": rer.normalize_method},
    }



@router.post("/reload_models")
def reload_models(db: Session = Depends(get_db), registry=Depends(get_registry)):
    registry.load_from_db(db)
    return {"message": "Active embedder and reranker reloaded from DB."}


@router.get("/settings", response_model=SettingsResponse)
def get_settings(db: Session = Depends(get_db)):
    setting = db.query(Setting).first()
    if not setting:
        raise AppException(ErrorCode.SETTINGS_NOT_FOUND, "No settings found", status_code=404)
    return SettingsResponse(
        id=setting.id,
        active_embedder_id=setting.active_embedder_id,
        active_reranker_id=setting.active_reranker_id,
        updated_at=str(setting.updated_at) if setting.updated_at else None,
    )


@router.put("/settings")
def update_settings(req: UpdateSettingsRequest, db: Session = Depends(get_db)):
    setting = db.query(Setting).first()
    if not setting:
        setting = Setting(active_embedder_id=req.active_embedder_id, active_reranker_id=req.active_reranker_id)
        db.add(setting)
    else:
        setting.active_embedder_id = req.active_embedder_id
        setting.active_reranker_id = req.active_reranker_id
    db.commit()
    return {"message": "Settings updated successfully"}

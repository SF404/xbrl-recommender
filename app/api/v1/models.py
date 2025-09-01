from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ...core.deps import get_db
from ...core.errors import AppException, ErrorCode
from ...schemas.reranker import RerankerRequest
from ...services.embedder_service import list_embedders, delete_embedder
from ...services.reranker_service import list_rerankers, get_reranker, delete_reranker

router = APIRouter()


@router.get("/embedders")
def get_embedders(db: Session = Depends(get_db)):
    return list_embedders(db)


@router.delete("/embedders/{embedder_id}")
def remove_embedder(embedder_id: int, db: Session = Depends(get_db)):
    obj = delete_embedder(db, embedder_id)
    if not obj:
        raise AppException(ErrorCode.NOT_FOUND, "Embedder not found", status_code=404)
    return {"message": "Embedder deleted successfully"}


@router.get("/rerankers")
def get_rerankers(db: Session = Depends(get_db)):
    return list_rerankers(db)


@router.patch("/rerankers/{reranker_id}")
def update_reranker(reranker_id: int, req: RerankerRequest, db: Session = Depends(get_db)):
    rer = get_reranker(db, reranker_id)
    if not rer:
        raise AppException(ErrorCode.NOT_FOUND, "Reranker not found", status_code=404)

    if req.name is not None:
        rer.name = req.name
    if req.version is not None:
        rer.version = req.version
    if req.path is not None:
        rer.path = req.path
    if req.normalize_method is not None:
        rer.normalize_method = req.normalize_method

    db.commit()
    db.refresh(rer)
    return {"message": "Reranker updated successfully.", "reranker_id": rer.id}


@router.delete("/rerankers/{reranker_id}")
def remove_reranker(reranker_id: int, db: Session = Depends(get_db)):
    obj = delete_reranker(db, reranker_id)
    if not obj:
        raise AppException(ErrorCode.NOT_FOUND, "Reranker not found", status_code=404)
    return {"message": "Reranker deleted successfully"}

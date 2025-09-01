from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from ...core.deps import get_db
from ...services.training import train_embedder, train_reranker

router = APIRouter()


@router.post("/embedder")
def train_embedder_endpoint(db: Session = Depends(get_db)):
    msg = train_embedder(db)
    return {"message": msg}


@router.post("/reranker")
def train_reranker_endpoint(db: Session = Depends(get_db)):
    msg = train_reranker(db)
    return {"message": msg}

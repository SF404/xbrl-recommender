from sqlalchemy.orm import Session
from ..models.entities import Reranker


def list_rerankers(db: Session):
    return db.query(Reranker).all()


def get_reranker(db: Session, rid: int):
    return db.query(Reranker).filter(Reranker.id == rid).first()


def delete_reranker(db: Session, rid: int):
    obj = get_reranker(db, rid)
    if obj:
        db.delete(obj)
        db.commit()
    return obj

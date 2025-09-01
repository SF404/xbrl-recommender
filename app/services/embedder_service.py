from sqlalchemy.orm import Session
from ..models.entities import Embedder


def list_embedders(db: Session):
    return db.query(Embedder).all()


def delete_embedder(db: Session, embedder_id: int):
    obj = db.query(Embedder).filter(Embedder.id == embedder_id).first()
    if obj:
        db.delete(obj)
        db.commit()
    return obj

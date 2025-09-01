from sqlalchemy.orm import Session
from ..models.entities import Setting


def get_current_setting(db: Session):
    return db.query(Setting).first()

from typing import List, Tuple
from sqlalchemy.orm import Session
from ..models.entities import Feedback


def load_feedback_pairs(db: Session) -> List[Tuple[str, str, int]]:
    data = []
    for r in db.query(Feedback).all():
        label = 1 if r.is_correct else 0
        data.append((r.query, r.reference, label))
    return data

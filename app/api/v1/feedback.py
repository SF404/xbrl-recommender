from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ...core.deps import get_db
from ...core.errors import AppException, ErrorCode
from ...schemas.feedback import FeedbackRequest, FeedbackResponse
from ...models.entities import Feedback, Taxonomy

router = APIRouter()


@router.get("")
def get_feedbacks(db: Session = Depends(get_db)):
    return db.query(Feedback).all()


@router.post("", response_model=FeedbackResponse)
def add_feedback(req: FeedbackRequest, db: Session = Depends(get_db)):
    taxonomy = db.query(Taxonomy).filter(Taxonomy.symbol == req.taxonomy).first()
    if not taxonomy:
        raise AppException(ErrorCode.NOT_FOUND, f"Taxonomy '{req.taxonomy}' not found", status_code=404)

    fb = Feedback(
        taxonomy_id=taxonomy.id,
        query=req.query,
        reference=req.reference,
        tag=req.tag,
        is_correct=req.is_correct,
        is_custom=req.is_custom,
        rank=req.rank,
    )
    db.add(fb)
    db.commit()
    return FeedbackResponse(message="Feedback recorded successfully.", saved=True)


@router.delete("/{feedback_id}")
def delete_feedback(feedback_id: int, db: Session = Depends(get_db)):
    fb = db.query(Feedback).filter(Feedback.id == feedback_id).first()
    if not fb:
        raise AppException(ErrorCode.NOT_FOUND, "Feedback not found", status_code=404)
    db.delete(fb)
    db.commit()
    return {"message": "Feedback deleted successfully"}


@router.patch("/{feedback_id}", response_model=FeedbackResponse)
def update_feedback(feedback_id: int, req: FeedbackRequest, db: Session = Depends(get_db)):
    fb = db.query(Feedback).filter(Feedback.id == feedback_id).first()
    if not fb:
        raise AppException(ErrorCode.NOT_FOUND, "Feedback not found", status_code=404)

    fb.query = req.query
    fb.reference = req.reference
    fb.tag = req.tag
    fb.is_correct = req.is_correct
    fb.is_custom = req.is_custom
    fb.rank = req.rank

    db.commit()
    db.refresh(fb)
    return FeedbackResponse(message="Feedback updated successfully.", saved=True)

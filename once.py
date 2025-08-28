from sqlalchemy.orm import Session
from app import Embedder, Reranker, Setting  # import your models
from app import SessionLocal  # import your DB session maker

db: Session = SessionLocal()

# Create default embedder
embedder = Embedder(
    name="all-MiniLM-L6-v2",
    path="sentence-transformers/all-MiniLM-L6-v2"
)

# Create default reranker
reranker = Reranker(
    name="cross-encoder/ms-marco-MiniLM-L-6-v2",
    path="cross-encoder/ms-marco-MiniLM-L-6-v2",
    normalize_method = "softmax"

)

db.add(embedder)
db.add(reranker)
db.commit()
db.refresh(embedder)
db.refresh(reranker)

# Create default settings pointing to above
settings = Setting(
    active_embedder_id=embedder.id,
    active_reranker_id=reranker.id
)

db.add(settings)
db.commit()

print("✅ Default embedder, reranker, and settings created.")

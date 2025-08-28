# app.py
# ============================================
# XBRL Tag Recommender API (organized version)
# ============================================

# -----------------------
# Standard Library
# -----------------------
import os
import json
import uuid
from io import BytesIO
from typing import List, Optional, Dict
from datetime import datetime
from contextlib import asynccontextmanager

# -----------------------
# Third-Party
# -----------------------
import numpy as np
import pandas as pd

from fastapi import FastAPI, Depends, BackgroundTasks, HTTPException, UploadFile, File
from pydantic import BaseModel

from sqlalchemy import (
    Column, Integer, String, Boolean, Text, ForeignKey, DateTime, create_engine
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker, Session
from sqlalchemy.sql import func

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

from sentence_transformers import CrossEncoder, SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

from tqdm import tqdm

# -----------------------
# Local
# -----------------------
from job_manager import jobs


# ============================================================
# 1) Configuration & Globals
# ============================================================
DATABASE_URL = "sqlite:///./app.db"   # swap with postgres://... in prod
INDEX_PATH   = "./data/FAISS_INDEX"
FINETUNE_DIR = "./models/finetuned"

Base = declarative_base()
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

index_cache: Dict[str, any] = {}
active_embeddings = None
active_reranker  = None


# ============================================================
# 2) ORM Models
# ============================================================
class Setting(Base):
    __tablename__ = "settings"
    id = Column(Integer, primary_key=True, index=True)
    active_embedder_id = Column(Integer, ForeignKey("embedders.id"))
    active_reranker_id = Column(Integer, ForeignKey("rerankers.id"))
    updated_at = Column(DateTime(timezone=True), server_default=func.now())

    embedder = relationship("Embedder", back_populates="settings")
    reranker = relationship("Reranker", back_populates="settings")


class Embedder(Base):
    __tablename__ = "embedders"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    version = Column(String)
    path = Column(Text)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    settings = relationship("Setting", back_populates="embedder")


class Reranker(Base):
    __tablename__ = "rerankers"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    version = Column(String)
    path = Column(Text)
    normalize_method = Column(String)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    settings = relationship("Setting", back_populates="reranker")


class Taxonomy(Base):
    __tablename__ = "taxonomies"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    symbol = Column(String)
    description = Column(Text)
    source_file = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    entries = relationship("TaxonomyEntry", back_populates="taxonomy")
    feedbacks = relationship("Feedback", back_populates="taxonomy")


class TaxonomyEntry(Base):
    __tablename__ = "taxonomy_entries"
    id = Column(Integer, primary_key=True, index=True)
    taxonomy_id = Column(Integer, ForeignKey("taxonomies.id"))
    tag = Column(String)
    datatype = Column(String)
    reference = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    taxonomy = relationship("Taxonomy", back_populates="entries")


class Feedback(Base):
    __tablename__ = "feedbacks"
    id = Column(Integer, primary_key=True, index=True)
    taxonomy_id = Column(Integer, ForeignKey("taxonomies.id"))
    query = Column(Text)
    reference = Column(Text)
    tag = Column(String)
    is_correct = Column(Boolean)
    rank = Column(Integer)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    taxonomy = relationship("Taxonomy", back_populates="feedbacks")


# ============================================================
# 3) DB Utilities
# ============================================================
def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ============================================================
# 4) Model Wrappers & Loaders
# ============================================================
class RerankerWrapper:
    def __init__(self, model_name, device="cpu", normalize_method="softmax"):
        self.model = CrossEncoder(model_name, device=device)
        self.normalize_method = normalize_method

    def rerank(self, query: str, docs: List[Document], top_k: int = 5):
        pairs = [(query, d.metadata["reference"]) for d in docs]
        raw_scores = self.model.predict(pairs)

        if self.normalize_method == "softmax":
            exp_scores = np.exp(raw_scores - np.max(raw_scores))
            scores = exp_scores / exp_scores.sum()
        elif self.normalize_method == "sigmoid":
            scores = 1 / (1 + np.exp(-raw_scores))
        elif self.normalize_method == "minmax":
            min_s, max_s = np.min(raw_scores), np.max(raw_scores)
            scores = (raw_scores - min_s) / (max_s - min_s) if max_s != min_s else np.ones_like(raw_scores) * 0.5
        else:
            scores = raw_scores

        scored = list(zip(docs, scores))
        return sorted(scored, key=lambda x: x[1], reverse=True)[:top_k]


def load_active_models(db: Session):
    """Load active embedder and reranker into memory (called once or after settings update)."""
    global active_embeddings, active_reranker

    setting = db.query(Setting).first()
    if not setting:
        raise RuntimeError("No settings found in DB. Please seed at least one embedder/reranker + setting.")

    # Load Embedder
    embedder = db.query(Embedder).filter(Embedder.id == setting.active_embedder_id).first()
    active_embeddings = HuggingFaceEmbeddings(model_name=embedder.path, model_kwargs={"device": "cpu"})

    # Load Reranker
    rer = db.query(Reranker).filter(Reranker.id == setting.active_reranker_id).first()
    active_reranker = RerankerWrapper(model_name=rer.path, normalize_method=rer.normalize_method)


# ============================================================
# 5) Vector Index Helpers
# ============================================================
def build_index_async(job_id: str, docs, embeddings, taxonomy: str, index_path: str):
    texts = [doc.page_content for doc in docs]
    metadatas = [doc.metadata for doc in docs]

    vectors = []
    total = len(docs)
    jobs[job_id] = {"status": "running", "progress": 0, "total": total, "done": 0}

    for i, doc in enumerate(tqdm(docs, desc=f"Embedding {taxonomy} docs")):
        vec = embeddings.embed_documents([doc.page_content])[0]
        vectors.append(vec)
        jobs[job_id]["done"] = i + 1
        jobs[job_id]["progress"] = int(((i + 1) / total) * 100)

    vs = FAISS.from_embeddings(
        text_embeddings=list(zip(texts, vectors)),
        embedding=embeddings,
        metadatas=metadatas
    )
    vs.save_local(f"{index_path}/{taxonomy}")
    jobs[job_id]["status"] = "completed"
    return vs


def load_index(taxonomy: str, embeddings):
    if taxonomy in index_cache:
        return index_cache[taxonomy]
    index_path = f"{INDEX_PATH}/{taxonomy}"
    vs = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    index_cache[taxonomy] = vs
    return vs


def load_feedback_pairs(db: Session):
    data = []
    records = db.query(Feedback).all()
    for r in records:
        label = 1 if r.is_correct else 0
        data.append((r.query, r.reference, label))
    return data


# ============================================================
# 6) Pydantic Schemas
# ============================================================
class BuildRequest(BaseModel):
    taxonomy: str


class QueryRequest(BaseModel):
    query: str
    taxonomy: str
    k: int = 5
    rerank: bool = True


class QueryResult(BaseModel):
    tag: str
    datatype: str
    reference: str
    score: float
    rank: int


class QueryResponse(BaseModel):
    query: str
    taxonomy: str
    results: List[QueryResult]
    # to validate the exception branch in /query
    error: Optional[str] = None


class FeedbackRequest(BaseModel):
    taxonomy: str
    query: str
    reference: str
    tag: str
    is_correct: bool
    rank: int


class FeedbackResponse(BaseModel):
    message: str
    saved: bool


class UpdateSettingsRequest(BaseModel):
    active_embedder_id: int
    active_reranker_id: int


class UploadTaxonomyRequest(BaseModel):
    symbol: str
    description: str
    sheet_name: str


class TaxonomyEntryRequest(BaseModel):
    tag: str
    datatype: str
    reference: str


class RerankerRequest(BaseModel):
    name: Optional[str] = None
    version: Optional[str] = None
    path: Optional[str] = None
    normalize_method: Optional[str] = None



# ============================================================
# 7) FastAPI App & Lifespan
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    db = SessionLocal()
    try:
        load_active_models(db)
        print("[Startup] Active embedder and reranker loaded.")
    except Exception as e:
        print(f"[Startup Warning] {e}")
    finally:
        db.close()

    yield
    print("[Shutdown] Cleaning up resources...")


app = FastAPI(title="MXBAI FAISS + Reranker API", lifespan=lifespan)
init_db()


# ============================================================
# 8) System / Settings Endpoints
# ============================================================
@app.post("/reload_models")
def reload_models(db: Session = Depends(get_db)):
    load_active_models(db)
    return {"message": "Active embedder and reranker reloaded from DB."}


@app.get("/settings")
def get_settings(db: Session = Depends(get_db)):
    setting = db.query(Setting).first()
    if not setting:
        raise HTTPException(status_code=404, detail="No settings found")
    return {
        "id": setting.id,
        "active_embedder_id": setting.active_embedder_id,
        "active_reranker_id": setting.active_reranker_id,
        "updated_at": setting.updated_at
    }


@app.put("/settings")
def update_settings(req: UpdateSettingsRequest, db: Session = Depends(get_db)):
    setting = db.query(Setting).first()
    if not setting:
        setting = Setting(
            active_embedder_id=req.active_embedder_id,
            active_reranker_id=req.active_reranker_id,
        )
        db.add(setting)
    else:
        setting.active_embedder_id = req.active_embedder_id
        setting.active_reranker_id = req.active_reranker_id
    db.commit()
    return {"message": "Settings updated successfully"}


# ============================================================
# 9) Index Build / Query Endpoints
# ============================================================
@app.post("/build_index")
def api_build_index(req: BuildRequest, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    job_id = str(uuid.uuid4())
    taxonomy = db.query(Taxonomy).filter(Taxonomy.symbol == req.taxonomy).first()

    docs = [
        Document(
            page_content=entry.reference,
            metadata={"tag": entry.tag, "datatype": entry.datatype, "reference": entry.reference}
        )
        for entry in taxonomy.entries
    ]
    background_tasks.add_task(build_index_async, job_id, docs, active_embeddings, req.taxonomy, INDEX_PATH)
    return {"message": "Index build started", "job_id": job_id}


@app.get("/status/{job_id}")
def get_status(job_id: str):
    return jobs.get(job_id, {"error": "Job not found"})


@app.post("/query", response_model=QueryResponse)
def api_query(req: QueryRequest):
    try:
        vectorstore = load_index(req.taxonomy, active_embeddings)
        if vectorstore is None:
            return QueryResponse(query=req.query, taxonomy=req.taxonomy, results=[])

        # ---- Guard against dimension mismatch ----
        query_vec = active_embeddings.embed_query(req.query)
        if query_vec is None:
            raise ValueError("Embedding model returned None")
        if vectorstore.index.d != len(query_vec):
            return QueryResponse(query=req.query, taxonomy=req.taxonomy, results=[])
        # ------------------------------------------

        docs = vectorstore.similarity_search_with_score(req.query, k=req.k * 5)

        if req.rerank:
            docs_only = [doc for doc, _ in docs]
            reranked = active_reranker.rerank(req.query, docs_only, top_k=req.k)
            results = [
                QueryResult(
                    tag=d.metadata["tag"], datatype=d.metadata["datatype"],
                    reference=d.metadata["reference"], score=float(score), rank=i + 1
                )
                for i, (d, score) in enumerate(reranked)
            ]
        else:
            results = [
                QueryResult(
                    tag=doc.metadata["tag"], datatype=doc.metadata["datatype"],
                    reference=doc.metadata["reference"], score=float(1 / (1 + score)), rank=i + 1
                )
                for i, (doc, score) in enumerate(docs[:req.k])
            ]

        return QueryResponse(query=req.query, taxonomy=req.taxonomy, results=results)

    except Exception as e:
        # catch FAISS assertion errors cleanly
        return QueryResponse(query=req.query, taxonomy=req.taxonomy, results=[], error=str(e))


from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses, CrossEncoder

# ============================================================
# 10) Training Endpoints
# ============================================================

@app.post("/train_embedder")
def api_train_embedder(db: Session = Depends(get_db)):
    pairs = load_feedback_pairs(db)
    if not pairs:
        return {"message": "No feedback data found."}

    # Create training examples based on feedback pairs (only positive ones)
    train_examples = [
        InputExample(texts=[q, ref], label=1.0) 
        for q, ref, lbl in pairs if lbl == 1
    ]

    # Initialize model and training settings
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    # Train the model
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=1,
        warmup_steps=10,
        show_progress_bar=True
    )

    # Save the trained model
    embedder_path = FINETUNE_DIR + "/embedder"
    model.save(embedder_path)

    # Get the latest version from DB
    latest_embedder = db.query(Embedder).order_by(Embedder.version.desc()).first()
    if latest_embedder and latest_embedder.version:
        new_version = int(latest_embedder.version.lstrip('v')) + 1
        new_version = f"v{new_version}"
    else:
        new_version = "v1"

    # Insert the new embedder entry
    embedder = Embedder(
        name="fine-tuned_embedder",
        version=new_version,
        path=embedder_path,
        is_active=True
    )
    db.add(embedder)
    db.commit()

    return {"message": f"Embedder fine-tuned and saved as {new_version}."}


from torch.utils.data import DataLoader
from sentence_transformers import InputExample

@app.post("/train_reranker")
def api_train_reranker(db: Session = Depends(get_db)):
    pairs = load_feedback_pairs(db)
    if not pairs:
        return {"message": "No feedback data found."}

    # Create training examples (use the feedback label as score)
    train_examples = [
        InputExample(texts=[q, ref], label=float(lbl)) for q, ref, lbl in pairs
    ]

    # Initialize CrossEncoder
    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", num_labels=1)

    # Create DataLoader from the training examples
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

    # Train the model
    model.fit(
        train_dataloader=train_dataloader,   # Pass DataLoader here
        epochs=5,
        warmup_steps=10,
        show_progress_bar=True,
    )

    # Save the trained model
    reranker_path = FINETUNE_DIR + "/reranker"
    model.save(reranker_path)

    # Get the latest version from DB or set to v1 if no reranker exists
    latest_reranker = db.query(Reranker).order_by(Reranker.version.desc()).first()

    if latest_reranker and latest_reranker.version:
        # Increment version if it exists
        new_version = int(latest_reranker.version.lstrip('v')) + 1
        new_version = f"v{new_version}"
    else:
        # Set to v1 if no reranker exists or version is None
        new_version = "v1"

    # Insert the new reranker entry into the DB
    reranker = Reranker(
        name="fine-tuned_reranker",
        version=new_version,
        path=reranker_path,
        normalize_method="softmax",
        is_active=True
    )
    db.add(reranker)
    db.commit()

    return {"message": f"Reranker fine-tuned and saved as {new_version}."}






# ============================================================
# 11) Embedders & Rerankers CRUD
# ============================================================
@app.get("/embedders")
def get_embedders(db: Session = Depends(get_db)):
    return db.query(Embedder).all()


@app.delete("/embedders/{embedder_id}")
def delete_embedder(embedder_id: int, db: Session = Depends(get_db)):
    embedder = db.query(Embedder).filter(Embedder.id == embedder_id).first()
    if not embedder:
        raise HTTPException(status_code=404, detail="Embedder not found")
    db.delete(embedder)
    db.commit()
    return {"message": "Embedder deleted successfully"}


@app.get("/rerankers")
def get_rerankers(db: Session = Depends(get_db)):
    return db.query(Reranker).all()

@app.patch("/rerankers/{reranker_id}")
def update_reranker(reranker_id: int, req: RerankerRequest, db: Session = Depends(get_db)):
    # Retrieve the reranker entry from the database
    reranker = db.query(Reranker).filter(Reranker.id == reranker_id).first()
    
    if not reranker:
        raise HTTPException(status_code=404, detail="Reranker not found")
    
    # Update the fields if they are provided in the request
    if req.name is not None:
        reranker.name = req.name
    if req.version is not None:
        reranker.version = req.version
    if req.path is not None:
        reranker.path = req.path
    if req.normalize_method is not None:
        reranker.normalize_method = req.normalize_method

    # Commit the changes to the database
    db.commit()
    db.refresh(reranker)

    return {"message": "Reranker updated successfully.", "reranker_id": reranker.id}


@app.delete("/rerankers/{reranker_id}")
def delete_reranker(reranker_id: int, db: Session = Depends(get_db)):
    reranker = db.query(Reranker).filter(Reranker.id == reranker_id).first()
    if not reranker:
        raise HTTPException(status_code=404, detail="Reranker not found")
    db.delete(reranker)
    db.commit()
    return {"message": "Reranker deleted successfully"}


# ============================================================
# 12) Taxonomy & Entries CRUD
# ============================================================
@app.post("/taxonomy/upload")
async def upload_taxonomy(
    file: UploadFile = File(...),
    meta: UploadTaxonomyRequest = Depends(),
    db: Session = Depends(get_db),
):
    # Read uploaded file into memory
    contents = await file.read()
    excel_data = BytesIO(contents)

    # Load Excel + validate sheet
    xl = pd.ExcelFile(excel_data)
    if meta.sheet_name not in xl.sheet_names:
        raise HTTPException(
            status_code=400,
            detail=f"Sheet '{meta.sheet_name}' not found in uploaded file. Available sheets: {xl.sheet_names}",
        )

    df = xl.parse(meta.sheet_name)

    # Validate required columns
    required_cols = {"tag", "type", "reference"}
    df.columns = df.columns.str.lower()
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required columns: {', '.join(missing_cols)}",
        )

    # Create taxonomy
    taxonomy = Taxonomy(
        name=meta.sheet_name,
        symbol=meta.symbol,
        description=meta.description,
        source_file=file.filename,
    )
    db.add(taxonomy)
    db.commit()
    db.refresh(taxonomy)

    # Add taxonomy entries
    for _, row in df.iterrows():
        entry = TaxonomyEntry(
            taxonomy_id=taxonomy.id,
            tag=row["tag"],
            datatype=row["type"],
            reference=row["reference"],
        )
        db.add(entry)
    db.commit()

    return {"message": "Taxonomy uploaded successfully", "taxonomy_id": taxonomy.id}


@app.get("/taxonomy/{taxonomy_id}")
def get_taxonomy(taxonomy_id: int, db: Session = Depends(get_db)):
    taxonomy = db.query(Taxonomy).filter(Taxonomy.id == taxonomy_id).first()
    if not taxonomy:
        raise HTTPException(status_code=404, detail="Taxonomy not found")
    return taxonomy


@app.delete("/taxonomy/{taxonomy_id}")
def delete_taxonomy(taxonomy_id: int, db: Session = Depends(get_db)):
    taxonomy = db.query(Taxonomy).filter(Taxonomy.id == taxonomy_id).first()
    if not taxonomy:
        raise HTTPException(status_code=404, detail="Taxonomy not found")
    db.delete(taxonomy)
    db.commit()
    return {"message": "Taxonomy deleted successfully"}


@app.get("/taxonomy/{taxonomy_id}/entries")
def get_taxonomy_entries(taxonomy_id: int, db: Session = Depends(get_db)):
    return db.query(TaxonomyEntry).filter(TaxonomyEntry.taxonomy_id == taxonomy_id).all()


@app.post("/taxonomy/{taxonomy_id}/entries")
def add_taxonomy_entry(taxonomy_id: int, req: TaxonomyEntryRequest, db: Session = Depends(get_db)):
    entry = TaxonomyEntry(
        taxonomy_id=taxonomy_id, tag=req.tag, datatype=req.datatype, reference=req.reference
    )
    db.add(entry)
    db.commit()
    return {"message": "Entry added successfully"}


@app.put("/taxonomy/entries/{entry_id}")
def update_taxonomy_entry(entry_id: int, req: TaxonomyEntryRequest, db: Session = Depends(get_db)):
    entry = db.query(TaxonomyEntry).filter(TaxonomyEntry.id == entry_id).first()
    if not entry:
        raise HTTPException(status_code=404, detail="Entry not found")
    entry.tag = req.tag
    entry.datatype = req.datatype
    entry.reference = req.reference
    db.commit()
    return {"message": "Entry updated successfully"}


@app.delete("/taxonomy/entries/{entry_id}")
def delete_taxonomy_entry(entry_id: int, db: Session = Depends(get_db)):
    entry = db.query(TaxonomyEntry).filter(TaxonomyEntry.id == entry_id).first()
    if not entry:
        raise HTTPException(status_code=404, detail="Entry not found")
    db.delete(entry)
    db.commit()
    return {"message": "Entry deleted successfully"}


# ============================================================
# 13) Feedback CRUD
# ============================================================
@app.get("/feedbacks")
def get_feedbacks(db: Session = Depends(get_db)):
    return db.query(Feedback).all()


@app.post("/feedback", response_model=FeedbackResponse)
def api_feedback(req: FeedbackRequest, db: Session = Depends(get_db)):
    # Note: original mixed name/symbol usage preserved intentionally
    taxonomy = db.query(Taxonomy).filter(Taxonomy.symbol == req.taxonomy).first()
    print(taxonomy)
    fb = Feedback(
        taxonomy_id=taxonomy.id, query=req.query, reference=req.reference,
        tag=req.tag, is_correct=req.is_correct, rank=req.rank
    )
    db.add(fb)
    db.commit()
    return FeedbackResponse(message="Feedback recorded successfully.", saved=True)


@app.delete("/feedbacks/{feedback_id}")
def delete_feedback(feedback_id: int, db: Session = Depends(get_db)):
    fb = db.query(Feedback).filter(Feedback.id == feedback_id).first()
    if not fb:
        raise HTTPException(status_code=404, detail="Feedback not found")
    db.delete(fb)
    db.commit()
    return {"message": "Feedback deleted successfully"}


@app.patch("/feedbacks/{feedback_id}", response_model=FeedbackResponse)
def update_feedback(feedback_id: int, req: FeedbackRequest, db: Session = Depends(get_db)):
    # Retrieve the feedback entry from the database
    fb = db.query(Feedback).filter(Feedback.id == feedback_id).first()
    
    if not fb:
        raise HTTPException(status_code=404, detail="Feedback not found")

    # Update the fields if they are provided in the request
    if req.query is not None:
        fb.query = req.query
    if req.reference is not None:
        fb.reference = req.reference
    if req.tag is not None:
        fb.tag = req.tag
    if req.is_correct is not None:
        fb.is_correct = req.is_correct
    if req.rank is not None:
        fb.rank = req.rank
    
    # Commit the changes to the database
    db.commit()
    db.refresh(fb)

    return FeedbackResponse(message="Feedback updated successfully.", saved=True)

import uuid
from fastapi import APIRouter, BackgroundTasks, Depends
from sqlalchemy.orm import Session
from langchain.schema import Document

from ...core.deps import get_db, get_registry
from ...core.errors import AppException, ErrorCode
from ...jobs.manager import job_get
from ...schemas.query import BuildRequest, QueryRequest, QueryResponse, QueryResult
from ...services.vectorstore import build_index_async, load_index
from ...services.taxonomy_service import get_taxonomy_by_symbol

router = APIRouter()


@router.post("/build_index")
def build_index(req: BuildRequest, background_tasks: BackgroundTasks, db: Session = Depends(get_db), registry=Depends(get_registry)):
    taxonomy = get_taxonomy_by_symbol(db, req.taxonomy)
    if not taxonomy:
        raise AppException(ErrorCode.NOT_FOUND, f"Taxonomy with symbol '{req.taxonomy}' not found.", status_code=404)

    docs = [
        Document(
            page_content=entry.reference,
            metadata={"tag": entry.tag, "datatype": entry.datatype, "reference": entry.reference},
        )
        for entry in taxonomy.entries
    ]
    job_id = str(uuid.uuid4())
    background_tasks.add_task(build_index_async, job_id, docs, registry.embedder, req.taxonomy)
    return {"message": "Index build started", "job_id": job_id}


@router.get("/status/{job_id}")
def get_status(job_id: str):
    return job_get(job_id, {"error": "Job not found"})


@router.post("", response_model=QueryResponse)
def query(req: QueryRequest, registry=Depends(get_registry)) -> QueryResponse:
    registry.ensure_loaded()

    # Load FAISS index
    try:
        vectorstore = load_index(req.taxonomy, registry.embedder)
    except Exception:
        raise AppException(ErrorCode.INDEX_NOT_FOUND, f"FAISS index for taxonomy '{req.taxonomy}' was not found.", status_code=404)

    # Guard against dimension mismatch
    q_vec = registry.embedder.embed_query(req.query)
    if q_vec is None:
        raise AppException(ErrorCode.MODEL_NOT_LOADED, "Embedder returned None for query embedding.", status_code=500)

    if vectorstore.index.d != len(q_vec):
        raise AppException(
            ErrorCode.DIMENSION_MISMATCH,
            f"Index dim ({vectorstore.index.d}) != embedder dim ({len(q_vec)}). Rebuild the index with the active embedder.",
            status_code=409,
        )

    # Similarity search (retrieve more if reranking)
    docs_with_scores = vectorstore.similarity_search_with_score(req.query, k=max(req.k * 5, req.k))
    if req.rerank:
        docs_only = [doc for doc, _ in docs_with_scores]
        reranked = registry.reranker.rerank(req.query, docs_only, top_k=req.k)
        results = [
            QueryResult(
                tag=d.metadata["tag"],
                datatype=d.metadata["datatype"],
                reference=d.metadata["reference"],
                score=float(score),
                rank=i + 1,
            )
            for i, (d, score) in enumerate(reranked)
        ]
    else:
        results = [
            QueryResult(
                tag=doc.metadata["tag"],
                datatype=doc.metadata["datatype"],
                reference=doc.metadata["reference"],
                score=float(1 / (1 + score)),
                rank=i + 1,
            )
            for i, (doc, score) in enumerate(docs_with_scores[:req.k])
        ]

    return QueryResponse(query=req.query, taxonomy=req.taxonomy, results=results)

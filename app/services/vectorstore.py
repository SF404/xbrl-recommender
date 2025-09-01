from typing import List, Dict
from tqdm import tqdm
from langchain.schema import Document
from langchain_community.vectorstores import FAISS

from ..jobs.manager import job_set, job_update
from ..core.config import get_settings

_index_cache: Dict[str, FAISS] = {}  # taxonomy symbol -> FAISS

def build_index_async(job_id: str, docs: List[Document], embeddings, taxonomy: str):
    """
    Embeds docs and saves FAISS index to disk, updating job status.
    """
    settings = get_settings()
    texts = [d.page_content for d in docs]
    metas = [d.metadata for d in docs]

    vectors = []
    total = len(docs)
    job_set(job_id, {"status": "running", "progress": 0, "total": total, "done": 0})

    for i, doc in enumerate(tqdm(docs, desc=f"Embedding {taxonomy} docs")):
        vec = embeddings.embed_documents([doc.page_content])[0]
        vectors.append(vec)
        job_update(job_id, done=i + 1, progress=int(((i + 1) / total) * 100))

    vs = FAISS.from_embeddings(
        text_embeddings=list(zip(texts, vectors)),
        embedding=embeddings,
        metadatas=metas,
    )
    vs.save_local(f"{settings.INDEX_PATH}/{taxonomy}")
    _index_cache[taxonomy] = vs
    job_update(job_id, status="completed")
    return vs


def load_index(taxonomy: str, embeddings):
    """
    Load FAISS index for the given taxonomy symbol; cache in memory.
    """
    if taxonomy in _index_cache:
        return _index_cache[taxonomy]

    settings = get_settings()
    index_path = f"{settings.INDEX_PATH}/{taxonomy}"
    vs = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    _index_cache[taxonomy] = vs
    return vs

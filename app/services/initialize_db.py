import os
import tempfile
import shutil
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sentence_transformers import SentenceTransformer, CrossEncoder
from app.db.session import SessionLocal, init_db
from app.models.entities import Embedder, Reranker, Setting
from app.core.config import get_settings
from pathlib import Path

# Initialize database
settings = get_settings()
DATABASE_URL = settings.DATABASE_URL

# GCP imports
from google.cloud import storage

engine_kwargs = {"pool_pre_ping": True}
engine = create_engine(DATABASE_URL, **engine_kwargs)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create and initialize the DB
init_db()

def _storage_client():
    if settings.GCS_CREDENTIALS_JSON:
        return storage.Client.from_service_account_json(settings.GCS_CREDENTIALS_JSON)
    return storage.Client()

def _get_bucket():
    client = _storage_client()
    return client.bucket(settings.GCS_BUCKET)

def _bucket_has_prefix(bucket, prefix: str) -> bool:
    blobs = list(bucket.list_blobs(prefix=prefix, max_results=1))
    return len(blobs) > 0

def _upload_directory(bucket, source_dir: str, dest_prefix: str):
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, source_dir)
            blob_path = f"{dest_prefix.rstrip('/')}/{rel_path.replace(os.sep, '/')}"
            bucket.blob(blob_path).upload_from_filename(full_path)

def _download_prefix_to_dir(bucket, prefix: str, dest_dir: str):
    for blob in bucket.list_blobs(prefix=prefix):
        # skip directory-like entries
        if blob.name.endswith("/"):
            continue
        rel_path = os.path.relpath(blob.name, prefix)
        target_path = os.path.join(dest_dir, rel_path)
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        blob.download_to_filename(target_path)

def download_and_save_model(model_id: str, save_path: str):
    """Download and save a model locally then upload to GCS bucket."""
    print(f"Downloading model: {model_id} to temporary dir then uploading to bucket")
    temp_dir = tempfile.mkdtemp()
    try:
        if "cross-encoder" in model_id or model_id.startswith("cross-encoder/"):
            model = CrossEncoder(model_id)
        else:
            model = SentenceTransformer(model_id)

        model.save(temp_dir)

        bucket = _get_bucket()
        dest_prefix = f"{settings.GCS_PREFIX}/{model_id.replace('/', '_')}"
        _upload_directory(bucket, temp_dir, dest_prefix)
        print(f"Uploaded model {model_id} to gs://{settings.GCS_BUCKET}/{dest_prefix}")
        return dest_prefix
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

def download_and_save_model_reranker(model_name: str, save_path: str):
    """Alias for reranker (cross-encoder) using GCS storage."""
    return download_and_save_model(model_name, save_path)

def initialize_models(db):
    # Define model identifiers and bucket prefixes
    embedder_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embedder_prefix = f"{settings.GCS_PREFIX}/{embedder_model_name.replace('/', '_')}"
    reranker_model_name = "cross-encoder/ms-marco-MiniLM-L6-v2"
    reranker_prefix = f"{settings.GCS_PREFIX}/{reranker_model_name.replace('/', '_')}"

    bucket = _get_bucket()

    # Download/upload embedder to bucket if missing
    if not _bucket_has_prefix(bucket, embedder_prefix):
        download_and_save_model(embedder_model_name, embedder_prefix)

    # Download/upload reranker to bucket if missing
    if not _bucket_has_prefix(bucket, reranker_prefix):
        download_and_save_model_reranker(reranker_model_name, reranker_prefix)

    # Ensure only one Setting entry exists
    existing_setting = db.query(Setting).first()
    if existing_setting:
        db.delete(existing_setting)
        db.commit()

    # Create Embedder and Reranker DB entries referencing the bucket prefix
    embedder = Embedder(name=embedder_model_name, version="v1", path=f"gs://{settings.GCS_BUCKET}/{embedder_prefix}", is_active=True)
    db.add(embedder)
    db.commit()

    reranker = Reranker(name=reranker_model_name, version="v1", path=f"gs://{settings.GCS_BUCKET}/{reranker_prefix}", normalize_method="softmax", is_active=True)
    db.add(reranker)
    db.commit()

    setting = Setting(active_embedder_id=embedder.id, active_reranker_id=reranker.id)
    db.add(setting)
    db.commit()

    return embedder, reranker, setting

def load_embedder_and_reranker(embedder_prefix_or_id: str, reranker_prefix_or_id: str):
    """Download model from GCS prefix to temp dir and load locally."""
    bucket = _get_bucket()

    def _download_and_load(prefix_or_id: str):
        # Accept both "gs://bucket/prefix" and "prefix" forms
        if prefix_or_id.startswith("gs://"):
            # strip gs://bucket/
            _, rest = prefix_or_id.split("gs://", 1)
            _, prefix = rest.split("/", 1)
        else:
            prefix = prefix_or_id

        temp_dir = tempfile.mkdtemp()
        _download_prefix_to_dir(bucket, prefix, temp_dir)
        # try loading as cross-encoder first if looks like cross-encoder
        model_obj = None
        try:
            if "cross-encoder" in prefix:
                model_obj = CrossEncoder(temp_dir)
            else:
                model_obj = SentenceTransformer(temp_dir)
        except Exception:
            # fallback
            try:
                model_obj = SentenceTransformer(temp_dir)
            except Exception:
                model_obj = CrossEncoder(temp_dir)
        return model_obj, temp_dir

    embedder, ed_temp = _download_and_load(embedder_prefix_or_id)
    reranker, rr_temp = _download_and_load(reranker_prefix_or_id)

    # Caller is responsible for keeping models in memory; we keep temp dirs until process exit
    return embedder, reranker

def initialize_db():
    db = SessionLocal()
    try:
        embedder, reranker, setting = initialize_models(db)

        # Load models into memory by downloading from GCS to temp dirs
        embedder_obj, reranker_obj = load_embedder_and_reranker(f"{settings.GCS_PREFIX}/{embedder.name.replace('/', '_')}", f"{settings.GCS_PREFIX}/{reranker.name.replace('/', '_')}")
        
        print("Database initialized with the embedder and reranker.")
    except Exception as e:
        print(f"Error initializing database: {e}")
    finally:
        db.close()


import os
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

# Create the database session and initialize the database if it doesn't exist
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create and initialize the DB if necessary
init_db()

def download_and_save_model(model_id: str, save_path: str):
    """Download and save a model. Uses CrossEncoder for cross-encoders."""
    print(f"Downloading and saving model: {model_id} to {save_path}")
    os.makedirs(save_path, exist_ok=True)

    if "cross-encoder" in model_id or model_id.startswith("cross-encoder/"):
        model = CrossEncoder(model_id)
    else:
        model = SentenceTransformer(model_id)

    # use the library's save to produce a loadable directory
    model.save(save_path)
    return save_path

def download_and_save_model_reranker(model_name: str, save_path: str):
    """Download and save the reranker (cross-encoder) model locally."""
    print(f"Downloading and saving reranker model: {model_name} to {save_path}")
    
    # Create the directory if it doesn't exist  
    os.makedirs(save_path, exist_ok=True)
    
    # Use CrossEncoder to load and save the model
    model = CrossEncoder(model_name)
    model.save(save_path)
    
    return save_path

def initialize_models(db):
    # Define model paths
    embedder_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embedder_save_path = f"./checkpoints/model/base/{embedder_model_name.replace('/', '_')}"  # Local path to save the embedder model
    reranker_model_name = "cross-encoder/ms-marco-MiniLM-L6-v2"
    reranker_save_path = f"./checkpoints/model/base/{reranker_model_name.replace('/', '_')}"  # Local path to save the reranker model

    # Download and save embedder if it doesn't exist locally
    if not os.path.exists(embedder_save_path):
        download_and_save_model(embedder_model_name, embedder_save_path)

    # Download and save reranker if it doesn't exist locally
    if not os.path.exists(reranker_save_path):
        download_and_save_model_reranker(reranker_model_name, reranker_save_path)

    # Ensure only one Setting entry exists
    existing_setting = db.query(Setting).first()
    if existing_setting:
        # If a setting already exists, delete it
        db.delete(existing_setting)
        db.commit()

    # Create Embedder and Reranker in the database
    embedder = Embedder(name=embedder_model_name, version="v1", path=embedder_save_path, is_active=True)
    db.add(embedder)
    db.commit()

    reranker = Reranker(name=reranker_model_name, version="v1", path=reranker_save_path, normalize_method="softmax", is_active=True)
    db.add(reranker)
    db.commit()

    # Create Settings Entry to link the active embedder and reranker
    setting = Setting(active_embedder_id=embedder.id, active_reranker_id=reranker.id)
    db.add(setting)
    db.commit()

    return embedder, reranker, setting

def load_embedder_and_reranker(embedder_path_or_id: str, reranker_path_or_id: str):
    """Load embedder (SentenceTransformer) and reranker (CrossEncoder) safely."""
    print(f"Loading embedder from: {embedder_path_or_id}")
    embedder = SentenceTransformer(embedder_path_or_id)

    print(f"Loading reranker from: {reranker_path_or_id}")
    # reranker should be CrossEncoder if it's a cross-encoder model
    if "cross-encoder" in reranker_path_or_id or reranker_path_or_id.startswith("cross-encoder/"):
        reranker = CrossEncoder(reranker_path_or_id)
    else:
        # if user actually saved a seq-class model, try loading with CrossEncoder anyway,
        # or fall back to SentenceTransformer only if intended.
        try:
            reranker = CrossEncoder(reranker_path_or_id)
        except Exception:
            reranker = SentenceTransformer(reranker_path_or_id)

    return embedder, reranker

def initialize_db():
    # Create a session instance using the correct syntax
    db = SessionLocal()  # Here we use the sessionmaker to create a session
    try:
        embedder, reranker, setting = initialize_models(db)

        # Load models into memory for fast use
        embedder, reranker = load_embedder_and_reranker(f"./models/{embedder.name.replace('/', '_')}", f"./models/{reranker.name.replace('/', '_')}")
        
        print("Database initialized with the embedder and reranker.")
        print(f"Embedder loaded with path: {embedder.model_name}, Reranker loaded with path: {reranker.model_name}")
    except Exception as e:
        print(f"Error initializing database: {e}")
    finally:
        db.close()  # Don't forget to close the session


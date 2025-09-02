from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from ..core.config import get_settings

settings = get_settings()

DATABASE_URL = settings.DATABASE_URL

engine_kwargs = {"pool_pre_ping": True}

engine = create_engine(DATABASE_URL, **engine_kwargs)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# ✅ Declare a single Base here
Base = declarative_base()

def init_db():
    """
    Import models so they register with Base.metadata, then create tables.
    Keep the import inside the function to avoid circular imports.
    """
    from ..models.entities import Setting, Embedder, Reranker, Taxonomy, TaxonomyEntry, Feedback
    Base.metadata.create_all(bind=engine)

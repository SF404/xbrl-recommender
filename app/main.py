from contextlib import asynccontextmanager
from typing import List
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .core.config import get_settings
from .core.logging import configure_logging
from .core.errors import install_exception_handlers
from .core.middleware import RequestContextMiddleware
from .db.session import init_db, SessionLocal
from .services.model_registry import ModelRegistry
from .api.router import api_router

ALLOWED_ORIGINS: List[str] = [
    "http://localhost:3000",
    "http://localhost:5174",
    "http://localhost:8000",
    "https://xbrl.briskbold.ai",
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    settings = get_settings()
    configure_logging("INFO")
    init_db()

    # Load active models at startup for fast responses
    registry = ModelRegistry()
    app.state.registry = registry
    try:
        db = SessionLocal()
        registry.load_from_db(db)
        print("[Startup] Active embedder and reranker loaded.")
    except Exception as e:
        # Non-fatal: app can still start; /reload_models can fix later
        print(f"[Startup Warning] {e}")
    finally:
        db.close()

    yield
    # Shutdown
    print("[Shutdown] Done.")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.APP_NAME, lifespan=lifespan)

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Middleware & errors
    app.add_middleware(RequestContextMiddleware)
    install_exception_handlers(app)

    # Routes
    app.include_router(api_router, prefix=settings.API_PREFIX)
    return app


app = create_app()

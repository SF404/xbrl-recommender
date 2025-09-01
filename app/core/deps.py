from typing import Generator
from fastapi import Request, Depends
from ..db.session import SessionLocal
from ..services.model_registry import ModelRegistry


def get_db() -> Generator:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_registry(request: Request) -> ModelRegistry:
    # set in app.state during startup
    return request.app.state.registry

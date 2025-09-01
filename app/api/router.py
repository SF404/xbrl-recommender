from fastapi import APIRouter
from .v1 import system, query, taxonomy, models, feedback, training

api_router = APIRouter()
api_router.include_router(system.router, prefix="/system", tags=["system"])
api_router.include_router(query.router, prefix="/query", tags=["query"])
api_router.include_router(taxonomy.router, prefix="/taxonomy", tags=["taxonomy"])
api_router.include_router(models.router, prefix="/models", tags=["models"])
api_router.include_router(feedback.router, prefix="/feedback", tags=["feedback"])
api_router.include_router(training.router, prefix="/training", tags=["training"])

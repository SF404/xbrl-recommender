from functools import lru_cache
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    APP_NAME: str = Field(default="XBRL Tag Recommender API")
    APP_ENV: str = Field(default="dev")  # "dev" or "prod"
    API_PREFIX: str = Field(default="/api/v1")

    DATABASE_URL: str = Field(default="sqlite:///./app.db")
    INDEX_PATH: str = Field(default="./data/FAISS_INDEX")
    FINETUNE_DIR: str = Field(default="./checkpoints/model/finetuned")
    DEVICE: str = Field(default="cpu")  # "cpu" or "cuda"

    class Config:
        env_file = ".env"
        case_sensitive = True

    @property
    def is_dev(self) -> bool:
        return self.APP_ENV.lower() == "dev"


@lru_cache()
def get_settings() -> Settings:
    return Settings()

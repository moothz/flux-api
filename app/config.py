import os
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """FLUX API Configuration Settings."""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 13005
    DEBUG_MODE: bool = False
    REQUIRE_API_KEY: bool = False
    API_KEY: Optional[str] = None

    # Model & Inference Settings
    MODEL_ID: str = "black-forest-labs/FLUX.1-schnell"
    QUANTIZATION: str = "nf4"  # 'nf4', 'fp8', or 'bf16'
    DEVICE: str = "cuda"
    TORCH_DTYPE: str = "bfloat16"

    # Default Generation Parameters
    DEFAULT_STEPS: int = 4
    DEFAULT_GUIDANCE_SCALE: float = 0.0
    DEFAULT_WIDTH: int = 1024
    DEFAULT_HEIGHT: int = 1024
    DEFAULT_STRENGTH: float = 0.60
    PREWARM: bool = True

    # Hugging Face Token (if accessing gated repos like FLUX.1-dev)
    HF_TOKEN: Optional[str] = None


settings = Settings()

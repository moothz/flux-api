from typing import Optional, Literal, List
from pydantic import BaseModel, Field


class GenerationRequest(BaseModel):
    """Text-to-Image Generation Request."""
    prompt: str = Field(..., description="Text prompt describing the image to generate")
    width: Optional[int] = Field(default=None, description="Image width in pixels (must be multiple of 16)")
    height: Optional[int] = Field(default=None, description="Image height in pixels (must be multiple of 16)")
    steps: Optional[int] = Field(default=None, ge=1, le=50, description="Number of diffusion inference steps")
    guidance_scale: Optional[float] = Field(default=None, ge=0.0, le=20.0, description="Guidance scale")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")
    response_format: Literal["binary", "b64_json"] = Field(default="binary", description="Response format")


class EditRequest(BaseModel):
    """Image-to-Image / Edit Request (JSON body format if image is provided as base64/URL)."""
    prompt: str = Field(..., description="Instruction or description for the edit")
    image_base64: Optional[str] = Field(default=None, description="Base64 encoded input image")
    image_url: Optional[str] = Field(default=None, description="URL of the input image")
    strength: Optional[float] = Field(default=None, ge=0.01, le=1.0, description="Transformation strength (0.1 = subtle, 0.9 = drastic)")
    steps: Optional[int] = Field(default=None, ge=1, le=50, description="Number of inference steps")
    guidance_scale: Optional[float] = Field(default=None, ge=0.0, le=20.0, description="Guidance scale")
    seed: Optional[int] = Field(default=None, description="Random seed")
    response_format: Literal["binary", "b64_json"] = Field(default="binary", description="Response format")


class ImageData(BaseModel):
    b64_json: Optional[str] = None
    url: Optional[str] = None


class OpenAIImageResponse(BaseModel):
    created: int
    data: List[ImageData]


class StatusResponse(BaseModel):
    status: str
    is_ready: bool
    model_id: str
    quantization: str
    device: str
    vram_allocated_mb: float
    vram_reserved_mb: float
    total_vram_mb: float

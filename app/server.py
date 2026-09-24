import base64
import io
import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Header,
    Request,
    Response,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import requests

from app.config import settings
from app.pipeline import flux_manager
from app.schemas import (
    EditRequest,
    GenerationRequest,
    OpenAIImageResponse,
    StatusResponse,
)

logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG_MODE else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("flux-api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting FLUX API server on port %d...", settings.PORT)
    flux_manager.load_model()
    flux_manager.prewarm()
    yield
    log.info("Shutting down FLUX API server...")


app = FastAPI(
    title="FLUX.1 Image Generation & Editing API",
    description="High-performance, lightweight REST API for Text-to-Image and Image-to-Image powered by FLUX.1 with 4-bit (NF4) quantization.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _verify_auth(authorization: Optional[str] = Header(None)):
    if not settings.REQUIRE_API_KEY:
        return
    if not authorization:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing Authorization header")
    token = authorization.replace("Bearer ", "").strip()
    if token != settings.API_KEY:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid API Key")


def _image_to_response(img: Image.Image, response_format: str = "binary") -> Response:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    if response_format == "b64_json":
        b64_str = base64.b64encode(img_bytes).decode("utf-8")
        return JSONResponse(
            content={
                "created": int(time.time()),
                "data": [{"b64_json": b64_str}],
            }
        )
    return Response(content=img_bytes, media_type="image/png")


def _load_image_from_bytes(data: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")


def _load_image_from_url_or_b64(url: Optional[str], b64: Optional[str]) -> Image.Image:
    if b64:
        if "," in b64:
            b64 = b64.split(",", 1)[1]
        try:
            raw = base64.b64decode(b64)
            return Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 image: {str(e)}")
    if url:
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            return Image.open(io.BytesIO(resp.content)).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to download image from URL: {str(e)}")
    raise HTTPException(status_code=400, detail="No image provided (must supply image file, base64, or URL)")


# ── Health & Status Endpoints ──────────────────────────────────────────────

@app.get("/health", response_model=StatusResponse, tags=["Status"])
@app.get("/api/v1/status", response_model=StatusResponse, tags=["Status"])
def health_check():
    alloc, reserved, total = flux_manager.get_vram_info()
    return StatusResponse(
        status="healthy" if flux_manager.is_ready else "initializing",
        is_ready=flux_manager.is_ready,
        model_id=settings.MODEL_ID,
        quantization=settings.QUANTIZATION,
        device=settings.DEVICE,
        vram_allocated_mb=alloc,
        vram_reserved_mb=reserved,
        total_vram_mb=total,
    )


# ── Text-to-Image Generation ──────────────────────────────────────────────

@app.post("/v1/images/generations", tags=["Generation"])
@app.post("/generate", tags=["Generation"])
async def generate_image(
    request: Request,
    prompt: Optional[str] = Form(None),
    width: Optional[int] = Form(None),
    height: Optional[int] = Form(None),
    steps: Optional[int] = Form(None),
    guidance_scale: Optional[float] = Form(None),
    seed: Optional[int] = Form(None),
    response_format: Optional[str] = Form(None),
    authorization: Optional[str] = Header(None),
):
    """Generate image from text prompt (JSON body or Multipart Form)."""
    _verify_auth(authorization)

    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        try:
            body = await request.json()
            p = body.get("prompt")
            w = body.get("width") or settings.DEFAULT_WIDTH
            h = body.get("height") or settings.DEFAULT_HEIGHT
            s = body.get("steps") or settings.DEFAULT_STEPS
            g = body.get("guidance_scale")
            sd = body.get("seed")
            fmt = body.get("response_format") or "binary"
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON payload: {str(e)}")
    else:
        p = prompt
        w = width or settings.DEFAULT_WIDTH
        h = height or settings.DEFAULT_HEIGHT
        s = steps or settings.DEFAULT_STEPS
        g = guidance_scale
        sd = seed
        fmt = response_format or "binary"

    if not p:
        raise HTTPException(status_code=400, detail="Field 'prompt' is required.")

    try:
        img = flux_manager.generate(
            prompt=p,
            width=w,
            height=h,
            steps=s,
            guidance_scale=g,
            seed=sd,
        )
        return _image_to_response(img, response_format=fmt)
    except Exception as e:
        log.exception("Generation error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# ── Image Editing / Img2Img ───────────────────────────────────────────────

@app.post("/v1/images/edits", tags=["Editing"])
@app.post("/edit", tags=["Editing"])
async def edit_image(
    request: Request,
    image: Optional[UploadFile] = File(None),
    prompt: Optional[str] = Form(None),
    strength: Optional[float] = Form(None),
    steps: Optional[int] = Form(None),
    guidance_scale: Optional[float] = Form(None),
    seed: Optional[int] = Form(None),
    response_format: Optional[str] = Form(None),
    authorization: Optional[str] = Header(None),
):
    """Edit or transform an image using a text instruction (Multipart Form or JSON body)."""
    _verify_auth(authorization)

    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        try:
            body = await request.json()
            p = body.get("prompt")
            st = body.get("strength") or settings.DEFAULT_STRENGTH
            s = body.get("steps") or settings.DEFAULT_STEPS
            g = body.get("guidance_scale")
            sd = body.get("seed")
            fmt = body.get("response_format") or "binary"
            img_b64 = body.get("image_base64") or body.get("image")
            img_url = body.get("image_url")
            init_img = _load_image_from_url_or_b64(img_url, img_b64)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON payload: {str(e)}")
    else:
        p = prompt
        st = strength or settings.DEFAULT_STRENGTH
        s = steps or settings.DEFAULT_STEPS
        g = guidance_scale
        sd = seed
        fmt = response_format or "binary"

        if image is not None:
            init_img = _load_image_from_bytes(await image.read())
        else:
            raise HTTPException(status_code=400, detail="Multipart field 'image' file is required.")

    if not p:
        raise HTTPException(status_code=400, detail="Field 'prompt' is required.")

    try:
        edited_img = flux_manager.edit(
            image=init_img,
            prompt=p,
            strength=st,
            steps=s,
            guidance_scale=g,
            seed=sd,
        )
        return _image_to_response(edited_img, response_format=fmt)
    except Exception as e:
        log.exception("Edit error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# ── Inpainting / Mask Editing ──────────────────────────────────────────────

@app.post("/v1/images/inpainting", tags=["Inpainting"])
@app.post("/inpaint", tags=["Inpainting"])
async def inpaint_image(
    image: UploadFile = File(...),
    mask_image: UploadFile = File(...),
    prompt: str = Form(...),
    strength: Optional[float] = Form(None),
    steps: Optional[int] = Form(None),
    guidance_scale: Optional[float] = Form(None),
    seed: Optional[int] = Form(None),
    response_format: Optional[str] = Form("binary"),
    authorization: Optional[str] = Header(None),
):
    """Inpaint a specific region of an image defined by a black/white mask."""
    _verify_auth(authorization)

    init_img = _load_image_from_bytes(await image.read())
    mask_img = _load_image_from_bytes(await mask_image.read()).convert("L")

    st = strength or 0.85
    s = steps or settings.DEFAULT_STEPS
    g = guidance_scale
    fmt = response_format or "binary"

    try:
        result_img = flux_manager.inpaint(
            image=init_img,
            mask_image=mask_img,
            prompt=prompt,
            strength=st,
            steps=s,
            guidance_scale=g,
            seed=seed,
        )
        return _image_to_response(result_img, response_format=fmt)
    except Exception as e:
        log.exception("Inpaint error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

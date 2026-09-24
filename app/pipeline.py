import io
import logging
import os
import time
from typing import Optional, Tuple
from PIL import Image
import torch

from app.config import settings

log = logging.getLogger(__name__)


class FluxManager:
    """Unified FLUX Pipeline Manager (Text2Image, Img2Img, Inpaint sharing VRAM)."""

    def __init__(self):
        self.pipe_t2i = None
        self.pipe_i2i = None
        self.pipe_inpaint = None
        self.is_ready = False

    def load_model(self):
        log.info(
            "Loading FLUX model %s on %s with quantization=%s...",
            settings.MODEL_ID,
            settings.DEVICE,
            settings.QUANTIZATION,
        )
        start_t = time.time()

        from diffusers import FluxPipeline, FluxImg2ImgPipeline, FluxInpaintPipeline, FluxTransformer2DModel
        from transformers import BitsAndBytesConfig

        hf_tok = settings.HF_TOKEN.strip() if settings.HF_TOKEN and settings.HF_TOKEN.strip() else None
        if not hf_tok and os.environ.get("HF_TOKEN") and os.environ.get("HF_TOKEN").strip():
            hf_tok = os.environ.get("HF_TOKEN").strip()
        token = hf_tok

        dtype = torch.bfloat16 if settings.TORCH_DTYPE == "bfloat16" else torch.float16

        if settings.QUANTIZATION.lower() == "nf4":
            log.info("Applying 4-bit (NF4) BitsAndBytes quantization to FLUX transformer...")
            quant_config_transformer = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
            )
            transformer = FluxTransformer2DModel.from_pretrained(
                settings.MODEL_ID,
                subfolder="transformer",
                quantization_config=quant_config_transformer,
                torch_dtype=dtype,
                token=token,
            )

            self.pipe_t2i = FluxPipeline.from_pretrained(
                settings.MODEL_ID,
                transformer=transformer,
                torch_dtype=dtype,
                token=token,
            )
            if settings.DEVICE == "cuda":
                self.pipe_t2i.to("cuda")
        elif settings.QUANTIZATION.lower() == "fp8":
            log.info("Loading FLUX in FP8 format...")
            self.pipe_t2i = FluxPipeline.from_pretrained(
                settings.MODEL_ID,
                torch_dtype=torch.float8_e4m3fn if hasattr(torch, "float8_e4m3fn") else dtype,
                token=token,
            )
            if settings.DEVICE == "cuda":
                self.pipe_t2i.to("cuda")
        else:
            log.info("Loading FLUX in standard %s format...", settings.TORCH_DTYPE)
            self.pipe_t2i = FluxPipeline.from_pretrained(
                settings.MODEL_ID,
                torch_dtype=dtype,
                token=token,
            )
            if settings.DEVICE == "cuda":
                self.pipe_t2i.to("cuda")

        # Share weights with Img2Img and Inpaint pipelines without duplicating VRAM
        log.info("Instantiating shared Img2Img and Inpaint pipelines...")
        self.pipe_i2i = FluxImg2ImgPipeline.from_pipe(self.pipe_t2i)
        self.pipe_inpaint = FluxInpaintPipeline.from_pipe(self.pipe_t2i)

        # Optimize VAE decoding memory
        try:
            self.pipe_t2i.enable_vae_slicing()
            self.pipe_t2i.enable_vae_tiling()
        except Exception:
            pass

        self.is_ready = True
        elapsed = time.time() - start_t
        log.info("FLUX model loaded successfully in %.2fs!", elapsed)

    def prewarm(self):
        if not settings.PREWARM or not self.is_ready:
            return
        log.info("Pre-warming FLUX pipeline with 1 step dummy generation...")
        try:
            self.generate(
                prompt="warmup test",
                width=512,
                height=512,
                steps=1,
                guidance_scale=0.0,
            )
            log.info("Pipeline pre-warmed successfully!")
        except Exception as e:
            log.warning("Warmup failed (non-critical): %s", e)

    def generate(
        self,
        prompt: str,
        width: Optional[int] = None,
        height: Optional[int] = None,
        steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> Image.Image:
        w = width or settings.DEFAULT_WIDTH
        h = height or settings.DEFAULT_HEIGHT
        s = steps or settings.DEFAULT_STEPS
        g = guidance_scale if guidance_scale is not None else settings.DEFAULT_GUIDANCE_SCALE

        # Ensure dimensions are multiples of 16 for FLUX VAE
        w = (w // 16) * 16
        h = (h // 16) * 16

        generator = None
        if seed is not None:
            generator = torch.Generator(device=settings.DEVICE).manual_seed(seed)

        start_t = time.time()
        result = self.pipe_t2i(
            prompt=prompt,
            width=w,
            height=h,
            num_inference_steps=s,
            guidance_scale=g,
            generator=generator,
        ).images[0]
        log.info("Generated image in %.2fs (steps=%d, size=%dx%d)", time.time() - start_t, s, w, h)
        return result

    def edit(
        self,
        image: Image.Image,
        prompt: str,
        strength: Optional[float] = None,
        steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> Image.Image:
        st = strength if strength is not None else settings.DEFAULT_STRENGTH
        s = steps or settings.DEFAULT_STEPS
        g = guidance_scale if guidance_scale is not None else settings.DEFAULT_GUIDANCE_SCALE

        # Resize image dimensions to multiples of 16
        orig_w, orig_h = image.size
        w = (orig_w // 16) * 16
        h = (orig_h // 16) * 16
        if (w, h) != (orig_w, orig_h):
            image = image.resize((w, h), Image.Resampling.LANCZOS)

        generator = None
        if seed is not None:
            generator = torch.Generator(device=settings.DEVICE).manual_seed(seed)

        start_t = time.time()
        result = self.pipe_i2i(
            prompt=prompt,
            image=image,
            strength=st,
            num_inference_steps=s,
            guidance_scale=g,
            generator=generator,
        ).images[0]
        log.info("Edited image in %.2fs (strength=%.2f, steps=%d)", time.time() - start_t, st, s)
        return result

    def inpaint(
        self,
        image: Image.Image,
        mask_image: Image.Image,
        prompt: str,
        strength: Optional[float] = None,
        steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> Image.Image:
        st = strength if strength is not None else 0.85
        s = steps or settings.DEFAULT_STEPS
        g = guidance_scale if guidance_scale is not None else settings.DEFAULT_GUIDANCE_SCALE

        orig_w, orig_h = image.size
        w = (orig_w // 16) * 16
        h = (orig_h // 16) * 16
        if (w, h) != (orig_w, orig_h):
            image = image.resize((w, h), Image.Resampling.LANCZOS)
            mask_image = mask_image.resize((w, h), Image.Resampling.NEAREST)

        generator = None
        if seed is not None:
            generator = torch.Generator(device=settings.DEVICE).manual_seed(seed)

        start_t = time.time()
        result = self.pipe_inpaint(
            prompt=prompt,
            image=image,
            mask_image=mask_image,
            strength=st,
            num_inference_steps=s,
            guidance_scale=g,
            generator=generator,
        ).images[0]
        log.info("Inpainted image in %.2fs (steps=%d)", time.time() - start_t, s)
        return result

    def get_vram_info(self) -> Tuple[float, float, float]:
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / (1024 * 1024)
            reserved = torch.cuda.memory_reserved() / (1024 * 1024)
            total = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
            return round(alloc, 2), round(reserved, 2), round(total, 2)
        return 0.0, 0.0, 0.0


flux_manager = FluxManager()

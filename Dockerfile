FROM python:3.11-slim

# Install system dependencies (git, ffmpeg, build tools, curl)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install PyTorch with CUDA 12.8 support (RTX 5090 / Blackwell sm_120)
RUN pip install --no-cache-dir \
    torch torchvision \
    --extra-index-url https://download.pytorch.org/whl/cu128

# Copy and install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY app/ /workspace/app/

# Environment defaults
ENV HOST=0.0.0.0
ENV PORT=13005
ENV PYTHONUNBUFFERED=1
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EXPOSE 13005

CMD ["python", "-m", "uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "13005"]

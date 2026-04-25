FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

# System deps for Docling (image processing, OCR fallbacks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# uv (fast Python package manager)
RUN pip install --no-cache-dir uv

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src/ ./src/

# Install into the system env; --system-site-packages keeps the CUDA torch from the base image
RUN uv pip install --system --no-cache-dir -e .

EXPOSE 8000

CMD ["enterprise-rag"]

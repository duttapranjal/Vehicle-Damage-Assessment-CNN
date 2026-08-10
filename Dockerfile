FROM python:3.10-slim

# System dependencies for OpenCV headless
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for Docker layer caching
COPY requirements.txt .

# Install Python dependencies
# tensorflow-cpu is ~300MB vs tensorflow's ~600MB — critical for free tier builds
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create runtime directories Flask needs
RUN mkdir -p static/uploads static/results assets/demo_images .hf_cache

# Render sets PORT env var automatically
ENV PORT=10000
ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=2

# gunicorn: 1 worker (TF model is large), 4 threads, 120s timeout for inference
CMD exec gunicorn app:app \
    --bind "0.0.0.0:$PORT" \
    --workers 1 \
    --threads 4 \
    --timeout 120 \
    --preload

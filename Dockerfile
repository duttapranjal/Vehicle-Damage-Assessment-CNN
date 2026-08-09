FROM python:3.10-slim

# System dependencies for OpenCV and TensorFlow
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create directories that Flask needs at runtime
RUN mkdir -p static/uploads static/results assets/demo_images .hf_cache

# Cloud Run sets PORT env var — gunicorn reads it
ENV PORT=8080
ENV PYTHONUNBUFFERED=1

# 1 worker because TF model is large; 120s timeout for inference
CMD exec gunicorn app:app \
    --bind "0.0.0.0:$PORT" \
    --workers 1 \
    --threads 4 \
    --timeout 120 \
    --preload

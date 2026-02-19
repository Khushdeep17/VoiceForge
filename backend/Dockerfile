FROM python:3.10

ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install system dependencies (audio)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements.txt .

# Upgrade pip
RUN pip install --upgrade pip

# Install CPU-only PyTorch explicitly
RUN pip install \
    torch torchaudio \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
RUN pip install -r requirements.txt

# Copy application code (models folder excluded via dockerignore)
COPY . .

# Create models directory (important for volume mount)
RUN mkdir -p /app/models

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

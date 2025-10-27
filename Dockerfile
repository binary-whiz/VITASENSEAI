FROM python:3.11-slim

# Keep Python output unbuffered (helps with logs)
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system deps needed by some packages (opencv, ffmpeg, etc.)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       libglib2.0-0 \
       libsm6 \
       libxrender1 \
       libxext6 \
       libgl1 \
       ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt ./
RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . /app

# Port that the app binds to (overrideable by platform)
ENV PORT=8080

# Use gunicorn as defined in requirements; bind to $PORT
CMD ["gunicorn", "--bind", ":8080", "--workers", "1", "--threads", "8", "--timeout", "0", "app:app"]

# Use lightweight Python 3.10 base image
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /workspace

# Install required system packages (for OpenCV, video processing, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git ffmpeg libsm6 libxext6 && \
    rm -rf /var/lib/apt/lists/*

# Copy Python dependencies
COPY requirements.txt ./

# Install dependencies (YOLOv8, torch CPU version, OpenCV, pandas, motmetrics, etc.)
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy all project files into container
COPY . .

# Default command (you can override in docker-compose.yml)
CMD ["python", "main.py", "--help"]

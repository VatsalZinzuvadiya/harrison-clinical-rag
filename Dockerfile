# Use official lightweight Python base image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1 \
    PORT=7860

WORKDIR /app

# Install build tools if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files and pre-indexed data
COPY . .

# Expose default port (7860 for Hugging Face Spaces)
EXPOSE 7860

# Start FastAPI server
CMD ["python", "api.py"]

FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy Python requirements and install them
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/

# Create a mount point for artifacts; this directory will be used to persist
# model checkpoints and other outputs. When running the trainer in Docker
# Compose, you can mount a host directory here.
RUN mkdir -p /app/artifacts

# Expose the port used by the FastAPI app
EXPOSE 8000

# Disable output buffering for Python to make logs appear in real time
ENV PYTHONUNBUFFERED=1

# Default command launches the API; the trainer overrides the command in
# docker-compose.yml when necessary
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]

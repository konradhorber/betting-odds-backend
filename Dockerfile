# Production Dockerfile
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry

# Configure poetry: don't create virtual environment, install only main deps
RUN poetry config virtualenvs.create false

# Copy poetry files
COPY pyproject.toml poetry.lock* ./

# Install production dependencies + google-cloud-storage for model download
RUN poetry install --only=main --no-dev --no-root && \
    pip install google-cloud-storage

# Copy application code
COPY src/ ./src/
COPY scripts/ ./scripts/

# Create models directory and make script executable
RUN mkdir -p models && chmod +x scripts/download_model.py

# Create non-root user
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/model-info || exit 1

# Set production environment and run model download then start FastAPI
ENV ENVIRONMENT=production
CMD ["sh", "-c", "python scripts/download_model.py && uvicorn src.main:app --host 0.0.0.0 --port 8000 --workers 1"]
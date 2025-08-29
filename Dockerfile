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

# Install production dependencies only
RUN poetry install --only=main --no-root

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

# Set production environment and run model download then start FastAPI
ENV ENVIRONMENT=production
CMD ["sh", "-c", "uvicorn src.main:app --host 0.0.0.0 --port 8000 --workers 1"]
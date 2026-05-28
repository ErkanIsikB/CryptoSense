# Use official Python 3.12 slim image as the base
FROM python:3.12-slim

# Set environment system variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies required for compilation and database connections
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy only the requirements file first to optimize Docker layer caching
COPY requirements.txt .

# Install Python package dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy the entire project codebase into the container
COPY . .

# Expose ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000 8501

# Default fallback command (can be overridden in docker-compose.yml)
CMD ["python", "main.py"]

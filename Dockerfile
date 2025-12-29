FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Copy requirements
COPY requirements_railway.txt .

# Install all dependencies together (avoids conflicts)
RUN pip install --no-cache-dir -r requirements_railway.txt

# Clear pip cache
RUN pip cache purge

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data static

# Clean up bytecode
RUN find /usr/local/lib/python3.11/site-packages -name "*.pyc" -delete && \
    find /usr/local/lib/python3.11/site-packages -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

EXPOSE 8000

# Start app (use multi_agent_app.py, not optimized_app.py)
CMD ["sh", "-c", "uvicorn multi_agent_app:app --host 0.0.0.0 --port ${PORT:-8000}"]
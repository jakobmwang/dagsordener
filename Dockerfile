FROM python:3.12-slim

WORKDIR /app

# Install uv for fast dependency management
RUN pip install uv

# Copy project files
COPY pyproject.toml uv.lock ./
COPY src/ ./src/
COPY completions.py knowledge.yaml ./

# Install dependencies
RUN uv pip install --system -e .

# Expose port
EXPOSE 8000

# Run with reload for development
CMD ["uvicorn", "src.search:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

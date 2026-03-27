FROM python:3.11-slim

WORKDIR /app

# System dependencies for compiling ML packages (if necessary)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install isolated requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all pre-compiled index structures and codebase
COPY . .

# Expose standard production ASIC port
EXPOSE 8000

# Fire the orchestrator engine
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

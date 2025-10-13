# Dockerfile
# Minimal, production-friendly image for FastAPI + your multi-agent app

FROM python:3.11-slim

# --- System setup (keep it lean) ---
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# (Optional) Faster, reproducible network for pip
# ENV PIP_DEFAULT_TIMEOUT=60

# --- Install Python deps first (layer caching) ---
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# --- Copy the rest of your code ---
COPY . .

# If you use "from src...." imports, ensure this exists:
# (repo root is on sys.path in Python, so "import src" works)
# If you chose a real package name (e.g., deal_intel), instead do: `pip install -e .`

# Expose FastAPI port
EXPOSE 8000

# Default runtime envs (override in CI/production as needed)
ENV USE_REDIS=true \
    CHAT_MODEL=disabled \
    SUMMARY_MODEL=disabled

# Start the API
# If your server module is at api/server.py and defines: app = FastAPI()
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]

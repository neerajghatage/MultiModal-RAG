FROM python:3.13-slim

WORKDIR /app

# System deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential libffi-dev libxml2-dev libxslt1-dev && \
    rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download ML models at build time from HuggingFace.
# Models are audited before each image build. At runtime, fastembed
# loads from this local cache — no internet access required.
RUN python -c "\
from fastembed import SparseTextEmbedding; \
from fastembed.rerank.cross_encoder.text_cross_encoder import TextCrossEncoder; \
SparseTextEmbedding(model_name='Qdrant/bm25'); \
TextCrossEncoder(model_name='Xenova/ms-marco-MiniLM-L-6-v2'); \
print('Models cached successfully')"

# App code
COPY . .

# Non-root user
RUN addgroup --system appgroup && \
    adduser --system --ingroup appgroup appuser && \
    chown -R appuser:appgroup /app
USER appuser

# NOTE: Do NOT copy .env into the image — pass secrets via
# docker-compose env_file or runtime environment variables.

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]

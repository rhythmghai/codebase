FROM python:3.13-slim

WORKDIR /app

# git is required at runtime, not just build time -- ingestion/ingest_from_url.py
# shells out to `git clone` for every POST /ingest call.
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=8000
EXPOSE 8000

# DATA_DIR and WORKSPACE_ROOT (see common/config.py) default to paths
# inside the image's own filesystem, which most container platforms treat
# as ephemeral -- mount a persistent volume at DATA_DIR (or override it to
# point at one) if ingested repos need to survive a container restart.
# This now includes the embedded Qdrant vector index (DATA_DIR/qdrant by
# default, see storage/vector_store.py) -- losing it means every repo has
# to be re-ingested to be queryable again, not just re-indexed for search.
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT}"]

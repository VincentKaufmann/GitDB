FROM python:3.12-slim AS base

LABEL maintainer="Vincent Kaufmann"
LABEL description="GitDB — GPU-accelerated version-controlled database"
LABEL org.opencontainers.image.source="https://github.com/VincentKaufmann/GitDB"

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./
COPY gitdb/ gitdb/

# Core install (torch + zstd only — fast, small)
RUN pip install --no-cache-dir .

# Default data directory
ENV GITDB_DATA=/data
VOLUME /data

EXPOSE 7474
EXPOSE 50051

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7474/api/health')" || exit 1

ENTRYPOINT ["gitdb"]
CMD ["serve", "--port", "7474", "--no-browser", "/data"]

# ── Full image with all optional deps ─────────────────────
FROM base AS full

RUN pip install --no-cache-dir ".[embed,ingest,crypto,cloud,grpc]"

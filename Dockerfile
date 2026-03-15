FROM python:3.12-slim AS base

LABEL maintainer="Vincent Kaufmann"
LABEL description="GitDB — GPU-accelerated version-controlled database"
LABEL org.opencontainers.image.source="https://github.com/VincentKaufmann/GitDB"

# System deps for torch CPU + zstd
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install GitDB with all optional deps except GPU
COPY pyproject.toml README.md ./
COPY gitdb/ gitdb/
RUN pip install --no-cache-dir ".[embed,ingest,crypto,cloud,grpc]"

# Default data directory
ENV GITDB_DATA=/data
VOLUME /data

# REST API port
EXPOSE 7474
# gRPC port
EXPOSE 50051

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7474/api/health')" || exit 1

# Default: start REST server
ENTRYPOINT ["gitdb"]
CMD ["serve", "--port", "7474", "--no-browser", "/data"]

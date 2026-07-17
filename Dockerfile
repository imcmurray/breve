FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY python ./python
COPY scenes ./scenes

RUN pip install --no-cache-dir -e ".[webai]"

ENV PYTHONUNBUFFERED=1
EXPOSE 8765

# Cloud hosts often set PORT
CMD ["sh", "-c", "breve-web --host 0.0.0.0 --port ${PORT:-8765}"]

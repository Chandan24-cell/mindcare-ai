FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       libgl1 \
       libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

ARG CACHE_BUST=2026-05-09-root-frontend
RUN echo "${CACHE_BUST}" > /tmp/mindcare-cache-bust

COPY . .
ARG CACHE_BUST=2026-05-09-v2
EXPOSE 7860

CMD ["python", "app.py"]

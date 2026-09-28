FROM python:3.11-slim

WORKDIR /app

# system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# copy and install python deps
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# copy source
COPY . /app

EXPOSE 8080

# Use gunicorn in production
CMD ["gunicorn", "serve_model:app", "-b", "0.0.0.0:8080", "--workers", "1"]

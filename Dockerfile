FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# (EXPOSE is optional for Cloud Run, but harmless)
EXPOSE 8080

# Use $PORT provided by Cloud Run; default to 8080 locally
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}"]

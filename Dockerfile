FROM python:3.13-rc-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .


# Use $PORT provided by Cloud Run; default to 8080 locally
CMD CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port 8090"]


# FROM python:3.13-rc-slim AS base

# ENV PYTHONDONTWRITEBYTECODE=1 \
#     PYTHONUNBUFFERED=1

# WORKDIR /app

# # Install dependencies
# COPY requirements.txt .
# RUN pip install --upgrade pip && pip install -r requirements.txt

# # Copy application code
# COPY backend/ /app/

# # Optional: If you want Nginx
# # COPY nginx.conf /etc/nginx/nginx.conf

# # Install Nginx
# # RUN apt-get update && apt-get install -y nginx && \
# #     apt-get clean && rm -rf /var/lib/apt/lists/*

# # Expose ports
# EXPOSE 80

# # Start both Nginx and Uvicorn using a process manager
# CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port 8000"]
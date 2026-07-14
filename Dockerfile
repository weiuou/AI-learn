FROM node:20-bookworm-slim AS frontend-build
WORKDIR /build/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_STATIC_ROOT=/app/frontend/dist
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
COPY --from=frontend-build /build/frontend/dist /app/frontend/dist
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "workspace_app.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

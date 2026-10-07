FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY eui_api.py .
COPY frontend ./frontend
COPY models_torch_range2_physics_01 ./models_torch_range2_physics_01

EXPOSE 8000

CMD ["sh", "-c", "uvicorn eui_api:app --host 0.0.0.0 --port ${PORT:-8000}"]

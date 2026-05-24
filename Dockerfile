FROM python:3.11-slim

WORKDIR /app
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --timeout 300 -r requirements.txt

COPY src/ ./src/
COPY data/ ./data/
COPY metrics/ ./metrics/

EXPOSE 7860
CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]

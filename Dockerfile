FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY models ./models
COPY data/processed ./data/processed
ENV PYTHONPATH=/app/src
EXPOSE 8000
CMD ["uvicorn", "marketing_intelligence.api:app", "--host", "0.0.0.0", "--port", "8000"]

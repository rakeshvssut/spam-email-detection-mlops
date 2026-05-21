FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY models ./models
COPY artifacts ./artifacts

ENV PYTHONPATH=/app
EXPOSE 8000

CMD ["python", "-m", "uvicorn", "src.deployment.api:app", "--host", "0.0.0.0", "--port", "8000"]

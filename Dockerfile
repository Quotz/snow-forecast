FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt requirements-api.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-api.txt

# Copy application
COPY . .

# Generate initial forecast data
RUN mkdir -p docs/history docs/verification

EXPOSE 8000

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]

FROM python:3.14-slim

# Set working directory
WORKDIR /app

# Install system dependencies
# postgresql-client for psql against the compose database. psycopg2-binary
# bundles libpq, so no compiler or headers.
RUN apt-get update && apt-get install -y --no-install-recommends \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Environment variables (can be overridden by docker-compose)
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=gavel
ENV DEBUG=true

# Expose port
EXPOSE 5000

# Run the application
CMD ["python", "runserver.py"]

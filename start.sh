#!/bin/bash
set -e

# Validate required environment variables
if [ -z "$DATABASE_URL" ] && [ -z "$DB_URI" ]; then
  echo "ERROR: DATABASE_URL or DB_URI environment variable must be set"
  echo "Format: postgresql://username:password@host:port/database"
  exit 1
fi

if [ -z "$SECRET_KEY" ]; then
  echo "ERROR: SECRET_KEY environment variable must be set"
  exit 1
fi

# Extract database connection details for health check
DB_URL="${DATABASE_URL:-$DB_URI}"
echo "Using database: $DB_URL"

# Wait for external PostgreSQL to be ready
echo "Waiting for external PostgreSQL database to be ready..."
for i in {1..60}; do
  if python -c "
import psycopg2
import sys
import os
try:
    conn = psycopg2.connect(os.environ.get('DATABASE_URL') or os.environ.get('DB_URI'))
    conn.close()
    sys.exit(0)
except Exception as e:
    sys.exit(1)
" 2>/dev/null; then
    echo "PostgreSQL database is ready!"
    break
  fi
  echo "Waiting for database... ($i/60)"
  sleep 2
done

# Final connection test
if ! python -c "
import psycopg2
import sys
import os
try:
    conn = psycopg2.connect(os.environ.get('DATABASE_URL') or os.environ.get('DB_URI'))
    conn.close()
    sys.exit(0)
except Exception as e:
    print(f'Failed to connect to database: {e}')
    sys.exit(1)
"; then
  echo "ERROR: Could not connect to external PostgreSQL database"
  echo "Please verify DATABASE_URL/DB_URI is correct and database is accessible"
  exit 1
fi

# Initialize Gavel database schema
echo "Initializing Gavel database schema..."
python initialize.py || true

# Start supervisor (which runs Gunicorn)
exec /usr/bin/supervisord -c /etc/supervisor/conf.d/supervisord.conf

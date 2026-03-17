#!/bin/bash
set -e

# Initialize PostgreSQL if needed
if [ ! -f /var/lib/postgresql/data/PG_VERSION ]; then
  echo "Initializing PostgreSQL database..."
  PG_BIN_INIT=$(ls -d /usr/lib/postgresql/*/bin | head -1)
  su - postgres -c "$PG_BIN_INIT/initdb -D /var/lib/postgresql/data"
  echo "PostgreSQL initialized"
fi

# Resolve PostgreSQL binary path (glob doesn't always expand)
PG_BIN=$(ls -d /usr/lib/postgresql/*/bin | head -1)

# Start PostgreSQL
echo "Starting PostgreSQL..."
su - postgres -c "$PG_BIN/pg_ctl -D /var/lib/postgresql/data -l /tmp/postgresql.log start"

# Wait for PostgreSQL to be ready
echo "Waiting for PostgreSQL to be ready..."
for i in {1..30}; do
  if su - postgres -c "psql -d postgres -c 'SELECT 1' > /dev/null 2>&1"; then
    echo "PostgreSQL is ready"
    break
  fi
  echo "Waiting for PostgreSQL... ($i/30)"
  sleep 1
done

# Verify gavel user and database exist
echo "Checking gavel database..."
if ! su - postgres -c "psql -d postgres -tAc \"SELECT 1 FROM pg_roles WHERE rolname='gavel'\"" | grep -q 1; then
  echo "Creating gavel user..."
  su - postgres -c "psql -d postgres -c \"CREATE USER gavel WITH PASSWORD 'gavel_prod_pass';\""
fi

if ! su - postgres -c "psql -d postgres -tAc \"SELECT 1 FROM pg_database WHERE datname='gavel'\"" | grep -q 1; then
  echo "Creating gavel database..."
  su - postgres -c "psql -d postgres -c \"CREATE DATABASE gavel OWNER gavel;\""
fi

# Initialize Gavel database
echo "Initializing Gavel database..."
python initialize.py || true

# Start supervisor
exec /usr/bin/supervisord -c /etc/supervisor/conf.d/supervisord.conf

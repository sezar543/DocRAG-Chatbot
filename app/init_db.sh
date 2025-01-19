#!/bin/bash
set -e

echo "Initializing the database..."

# Ensure the data directory has the correct permissions
if [ -d "/var/lib/postgresql/data/pgdata" ]; then
    chown -R postgres:postgres /var/lib/postgresql/data/pgdata
else
    echo "Data directory /var/lib/postgresql/data/pgdata does not exist."
    exit 1
fi

# Use the environment variables
PGUSER=${POSTGRES_USER}
PGPASSWORD=${POSTGRES_PASSWORD}
PGDATABASE=${POSTGRES_DB}
PGPORT=${DB_PORT}

# Export the environment variables for psql
export PGUSER PGPASSWORD PGDATABASE PGPORT

# Check if the database exists and create if it doesn't
if [ $(psql -tAc "SELECT 1 FROM pg_database WHERE datname='${PGDATABASE}'") != '1' ]; then
    echo "Creating database and user..."
    psql -v ON_ERROR_STOP=1 <<EOSQL
        DO
        \$do\$
        BEGIN
            IF NOT EXISTS (
                SELECT
                FROM   pg_catalog.pg_roles
                WHERE  rolname = '${PGUSER}') THEN

                CREATE ROLE ${PGUSER} WITH SUPERUSER LOGIN PASSWORD '${PGPASSWORD}';
            END IF;
        END
        \$do\$;

        DO
        \$do\$
        BEGIN
            IF NOT EXISTS (
                SELECT
                FROM   pg_catalog.pg_database
                WHERE  datname = '${PGDATABASE}') THEN

                CREATE DATABASE ${PGDATABASE} OWNER ${PGUSER};
            END IF;
        END
        \$do\$;
EOSQL
else
    echo "Database already exists, skipping initialization."
fi

echo "Database initialization complete."
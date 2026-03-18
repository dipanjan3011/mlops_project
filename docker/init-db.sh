#!/bin/bash
set -e

# Create a separate database for Airflow so its alembic migrations
# don't conflict with MLflow's migrations in the 'mlops' database.
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" <<-EOSQL
    CREATE DATABASE airflow;
    GRANT ALL PRIVILEGES ON DATABASE airflow TO $POSTGRES_USER;
EOSQL

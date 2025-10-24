#!/bin/bash
set -e  # Exit immediately if any command fails

echo " Setting up local environment..."

# Step 1: Check for Docker
if ! command -v docker &> /dev/null
then
    echo "Docker is not installed. Please install Docker first."
    exit 1
fi

# Step 2: Pull the TimescaleDB image (latest stable)
echo "Pulling TimescaleDB image..."
docker pull timescale/timescaledb-ha:pg17

# Step 3: Start the TimescaleDB container using Docker Compose
echo "Starting TimescaleDB container..."
docker compose up -d

# Step 4: Wait for TimescaleDB to be ready
echo "Waiting for TimescaleDB to be ready..."
sleep 5

# Step 5: (Optional) Run SQL init commands if needed
echo "Creating tables and hypertables..."
docker exec -i timescaledb psql -U admin -d mytimescale <<'SQL'
CREATE TABLE IF NOT EXISTS sensor_data (
    time TIMESTAMPTZ NOT NULL,
    sensor_id TEXT,
    value DOUBLE PRECISION
);
SELECT create_hypertable('sensor_data', 'time', if_not_exists => TRUE);
SQL

echo "TimescaleDB setup complete!"

# Step 6: Run Prefect flow (optional)
# echo "Running Prefect flow..."
# prefect deployment run 'etl-flow'

echo "Environment is ready to use!"

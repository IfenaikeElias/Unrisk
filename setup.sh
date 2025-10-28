#!/bin/bash
set -e  # Exit if any command fails

echo "Setting up local environment..."

# Check for Docker
if ! command -v docker &> /dev/null
then
    echo "Docker is not installed. Please install Docker first."
    exit 1
fi

# Pull TimescaleDB
echo "Pulling TimescaleDB image..."
docker pull timescale/timescaledb-ha:pg17

# Start TimescaleDB container
echo "Starting TimescaleDB container..."
docker compose up -d

# Wait for DB
echo "Waiting for TimescaleDB to be ready..."
sleep 5

# Create market_data table & hypertable
echo "Creating market_data table and hypertable..."
docker exec -i timescaledb psql -U admin -d mytimescale <<'SQL'
CREATE TABLE IF NOT EXISTS market_data (
    time TIMESTAMPTZ NOT NULL,
    ticker TEXT NOT NULL,
    open DOUBLE PRECISION,
    high DOUBLE PRECISION,
    low DOUBLE PRECISION,
    close DOUBLE PRECISION,
    adj_close DOUBLE PRECISION,
    volume BIGINT,
    returns DOUBLE PRECISION,
    momentum DOUBLE PRECISION,
    volatility DOUBLE PRECISION,
    infl_adj_return DOUBLE PRECISION,
    bullish_momentum DOUBLE PRECISION,
    vol_unemp_risk DOUBLE PRECISION,
    gdp DOUBLE PRECISION,
    inflation DOUBLE PRECISION,
    unemployment DOUBLE PRECISION,
    PRIMARY KEY (time, ticker)
);
SELECT create_hypertable('market_data', 'time', if_not_exists => TRUE);
SQL

# Run Python script to seed last 1+ year of stock data (max of 2 years) for HRP model to work with 
echo "Seeding TimescaleDB with historical data..."
python3 src/seed.py

echo "TimescaleDB setup complete with one year of historical stock data for chosen tickers!"
import datetime as dt
from src.Data_injestion import Injest
import pandas as pd
from sqlalchemy import create_engine
from utils.helper_functions import wide_to_long 

# DB URL
DB_URL = "postgresql+psycopg2://admin:password@localhost:5432/stockDB"
# Get last 1+ year of data
end_date = dt.datetime.now()
start_date = end_date - dt.timedelta(days=365)

# Load tickers
tickers = {
    "stock_ticker": ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'JNJ', 'PFE', 'JPM', 'GS', 'XOM', 'CVX'],
    "ETF_ticker": ['XLK', 'XLV', 'XLF', 'XLE'],
}

# Initialize injestor
injestor = Injest(tickers=tickers, end_date=end_date, up_to=(365*2))

# Aggregate wide data
df_wide = injestor.aggregate()

# Convert to long format (pass all tickers)
all_tickers = tickers["stock_ticker"] + tickers["ETF_ticker"]
df_long = wide_to_long(df_wide, all_tickers)

# Insert into TimescaleDB
engine = create_engine(DB_URL)
df_long.to_sql("market_data", engine, if_exists="append", index=False)
engine.dispose()

print(f"Seeded {len(df_long)} rows into market_data")
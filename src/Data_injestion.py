import datetime as dt
import yfinance as yf
import pandas as pd
import pandas_datareader.data as web
import time
from typing import Dict, List
from pathlib import Path
from utils.helper_functions import compute_volatility, compute_momentum, compute_return
from dateutil.relativedelta import relativedelta

root = Path().resolve().parent  

class Injest:
    def __init__(self, tickers: Dict[str, List[str]], end_date: dt.datetime= dt.datetime.today(),  up_to: int = 6):
        self._tickers = tickers
        self._end = end_date
        self._up_to = up_to
        self._start_val = self._end - dt.timedelta(days= self._up_to)

    def process_date(self, data: pd.DataFrame, start) -> pd.DataFrame:
        date_range = pd.date_range(start=start, end=self._end, freq="D")
        data = data.reindex(date_range).ffill()

        return data
    
    @property
    def _start(self):
        return self._start_val
    # we can use self._start everywhere now

    @_start.setter
    def _start(self, new_start):
        self._start_val = new_start
   
    def fetch_stock_data(self, max_retries: int = 3, retry_delay: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
        
        start = self._start
        
        for attempt in range(max_retries):
            try:
                # Fetch data
                stock_data = yf.download(
                    self._tickers["stock_ticker"], 
                    start=start, 
                    end=self._end, 
                    interval="1d", 
                    auto_adjust=False
                )
                ETF_data = yf.download(
                    self._tickers["ETF_ticker"], 
                    start=start, 
                    end=self._end, 
                    interval="1d", 
                    auto_adjust=False
                )
                
                # Null handling - check if data is empty
                if stock_data.empty:
                    raise ValueError("Stock data is empty. No data returned from yfinance.")
                if ETF_data.empty:
                    raise ValueError("ETF data is empty. No data returned from yfinance.")
                
                # Check for excessive nulls (more than 50% missing)
                stock_null_pct = (stock_data.isnull().sum().sum() / (stock_data.shape[0] * stock_data.shape[1])) * 100
                etf_null_pct = (ETF_data.isnull().sum().sum() / (ETF_data.shape[0] * ETF_data.shape[1])) * 100
                
                if stock_null_pct > 50:
                    raise ValueError(f"Stock data has {stock_null_pct:.2f}% null values. Data quality too low.")
                if etf_null_pct > 50:
                    raise ValueError(f"ETF data has {etf_null_pct:.2f}% null values. Data quality too low.")
                
                # Process columns
                stock_data.columns = ['_'.join(col).strip() for col in stock_data.columns]
                ETF_data.columns = ['_'.join(col).strip() for col in ETF_data.columns]
                
                # Resample and forward fill
                stock_data = self.process_date(stock_data.resample('D').ffill(), start).ffill()
                ETF_data = self.process_date(ETF_data.resample('D').ffill(), start).ffill()
                
                return ETF_data, stock_data
                
            except Exception as e:
                print(f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}")
                
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    print("Max retries reached. Raising exception.")
                    raise RuntimeError(f"Failed to fetch stock data after {max_retries} attempts: {str(e)}")

    def fetch_econ_data(self, max_retries: int = 3, retry_delay: int = 5) -> pd.DataFrame:
        
        start = self._start
        econ_start = self._end - relativedelta(months=8)  # Use local variable instead of modifying self._start
        
        for attempt in range(max_retries):
            try:
                # Fetch data
                gdp_data = web.DataReader('GDP', 'fred', econ_start, self._end)
                unemployment_data = web.DataReader('UNRATE', 'fred', econ_start, self._end)
                inflation = web.DataReader('CPIAUCSL', 'fred', econ_start, self._end)
                
                # Null handling - check if data is empty
                if gdp_data.empty:
                    raise ValueError("GDP data is empty. No data returned from FRED.")
                if unemployment_data.empty:
                    raise ValueError("Unemployment data is empty. No data returned from FRED.")
                if inflation.empty:
                    raise ValueError("Inflation data is empty. No data returned from FRED.")
                
                # Check for excessive nulls before resampling
                gdp_null_pct = (gdp_data.isnull().sum().sum() / gdp_data.shape[0]) * 100
                unemp_null_pct = (unemployment_data.isnull().sum().sum() / unemployment_data.shape[0]) * 100
                infl_null_pct = (inflation.isnull().sum().sum() / inflation.shape[0]) * 100
                
                if gdp_null_pct > 50:
                    raise ValueError(f"GDP data has {gdp_null_pct:.2f}% null values. Data quality too low.")
                if unemp_null_pct > 50:
                    raise ValueError(f"Unemployment data has {unemp_null_pct:.2f}% null values. Data quality too low.")
                if infl_null_pct > 50:
                    raise ValueError(f"Inflation data has {infl_null_pct:.2f}% null values. Data quality too low.")
                
                # Resample to daily frequency
                gdp_data = gdp_data.resample('D').ffill()
                unemployment_data = unemployment_data.resample('D').ffill()
                inflation = inflation.resample('D').ffill()
                
                # Process date range
                gdp_data = self.process_date(gdp_data, start)
                unemployment_data = self.process_date(unemployment_data, start)
                inflation = self.process_date(inflation, start)
                
                # Concatenate
                econ = pd.concat([
                    gdp_data.rename(columns={"GDP": "gdp"}),
                    unemployment_data.rename(columns={"UNRATE": "unemployment"}),
                    inflation.rename(columns={"CPIAUCSL": "inflation"})
                ], axis=1)
                
                # Slice to date range and forward fill
                econ = econ.loc[start: self._end].ffill()
                return econ
                
            except Exception as e:
                print(f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}")
                
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    print("Max retries reached. Raising exception.")
                    raise RuntimeError(f"Failed to fetch economic data after {max_retries} attempts: {str(e)}")
        
    @staticmethod
    def createfeatures(df, tickers):
        df = compute_volatility(df, 5, tickers)
        df = compute_momentum(df, 5, tickers)
        df = compute_return(df, tickers)
        for ticker in tickers:
            df[f"{ticker}_infl_adj_return"]    = df[f"{ticker}_returns"] * df["inflation"]
            df[f"{ticker}_bullish_momentum"]  = df[f"{ticker}_momentum"] * df["gdp"]
            df[f"{ticker}_vol_unemp_risk"]    = df[f"{ticker}_volatility"] * df["unemployment"]
        return df


    def aggregate(self)-> pd.DataFrame:
        stock_data, ETF_data = self.fetch_stock_data()
        print(stock_data.tail())
        econ = self.fetch_econ_data()
        general_data = pd.concat([stock_data, ETF_data, econ], axis= 1)
        tickers = self._tickers["stock_ticker"] + self._tickers["ETF_ticker"]
        general_data = self.createfeatures(general_data, tickers).ffill().bfill()
        return general_data

if __name__ == "__main__":
    # Usage for local runs only. Wrapped under main guard to avoid
    # execution during imports (e.g., from Prefect flows), also additional
    # save to csv for quick checking... :)
    tickers = {
        "stock_ticker": [
            'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'JNJ', 'PFE', 'JPM', 'GS', 'XOM', 'CVX'
        ],
        "ETF_ticker": ['XLK', 'XLV', 'XLF', 'XLE']
    }
    injestor = Injest(tickers, dt.datetime(2025, 6, 1))
    df = injestor.aggregate()
    df.to_csv(f"{root}/data/clean_data/df.csv", index=True) 
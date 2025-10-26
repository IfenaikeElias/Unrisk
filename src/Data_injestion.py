import datetime as dt
from dateutil.relativedelta import relativedelta
import yfinance as yf
import pandas as pd
import numpy as np
import pandas_datareader.data as web
import sys
from typing import Dict, List
from pathlib import Path
from utils.helper_functions import compute_volatility, compute_momentum, compute_return

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
   
    def fetch_stock_data(self) -> pd.DataFrame:
        start = self._start
        stock_data = yf.download(self._tickers["stock_ticker"], start=self._start, end=self._end, interval= "1d", auto_adjust=False)
        ETF_data = yf.download(self._tickers["ETF_ticker"], start=self._start, end=self._end, interval= "1d", auto_adjust=False)
        stock_data.columns = ['_'.join(col).strip() for col in stock_data.columns]
        ETF_data.columns = ['_'.join(col).strip() for col in ETF_data.columns]
        stock_data = self.process_date(stock_data.resample('D').ffill(), start).ffill()
        ETF_data = self.process_date(ETF_data.resample('D').ffill(), start).ffill()


        return ETF_data, stock_data

    def fetch_econ_data(self) -> pd.DataFrame:
        start = self._start
        self._start = self._end - relativedelta(months= 8)
        gdp_data = web.DataReader('GDP', 'fred', self._start, self._end)
        unemployment_data = web.DataReader('UNRATE', 'fred', self._start, self._end)
        inflation = web.DataReader('CPIAUCSL', 'fred', self._start, self._end)
        gdp_data = gdp_data.resample('D').ffill()
        unemployment_data = unemployment_data.resample('D').ffill()
        inflation = inflation.resample('D').ffill()        
        gdp_data = self.process_date(gdp_data, start)
        unemployment_data = self.process_date(unemployment_data, start)
        inflation = self.process_date(inflation, start)
        econ = pd.concat([
            gdp_data.rename(columns={"GDP": "gdp"}),
            unemployment_data.rename(columns={"UNRATE": "unemployment"}),
            inflation.rename(columns={"CPIAUCSL": "inflation"})
        ], axis=1)
        econ = econ.loc[start: self._end].ffill()
        econ.index.name = "Date"

        return econ
    
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
    # Example usage for local runs only. Wrapped under main guard to avoid
    # execution during imports (e.g., from Prefect flows).
    tickers = {
        "stock_ticker": [
            'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'JNJ', 'PFE', 'JPM', 'GS', 'XOM', 'CVX'
        ],
        "ETF_ticker": ['XLK', 'XLV', 'XLF', 'XLE']
    }
    injestor = Injest(tickers, dt.datetime(2025, 6, 1))
    df = injestor.aggregate()
    df.to_csv(f"{root}/data/clean_data/df.csv", index=True) 
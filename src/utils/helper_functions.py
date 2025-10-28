import numpy as np
import pandas as pd
from typing import List
from torch.optim import Adam, RMSprop
import optuna
from sqlalchemy import create_engine
from typing import Optional



def compute_volatility(df: pd.DataFrame, period: int, tickers: List[str]) -> pd.DataFrame:
    out = df.copy()
    for ticker in tickers:
        price_col = f"Adj Close_{ticker}"
        r = np.log(df[price_col] / df[price_col].shift(1)).fillna(0)
        vol = r.rolling(window=period).std() * np.sqrt(period)
        out[f"{ticker}_volatility"] = vol.fillna(0)

    return out


def compute_momentum(df, period, tickers):
    out = df.copy()
    for ticker in tickers:
        price_col = f"Adj Close_{ticker}"
        mom = np.log(df[price_col] / df[price_col].shift(period)).fillna(0)
        out[f"{ticker}_momentum"] = mom

    return out


def compute_return(df, tickers):
    out = df.copy()
    for ticker in tickers:
        price_col = f"Adj Close_{ticker}"
        ret = np.log(df[price_col] / df[price_col].shift(1)).fillna(0)
        out[f"{ticker}_returns"] = ret
    return out

def wide_to_long(df: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    """
    Convert wide format DataFrame to long format for TimescaleDB insertion.
    """
    
    # df: Wide format DataFrame with multi-ticker columns
    # we'll get a long format DataFrame with one row per ticker per timestamp
    long_data = []
    
    for ticker in tickers:
        ticker_df = pd.DataFrame({
            'time': df.index,
            'ticker': ticker,
            'open': df.get(f'Open_{ticker}'),
            'high': df.get(f'High_{ticker}'),
            'low': df.get(f'Low_{ticker}'),
            'close': df.get(f'Close_{ticker}'),
            'adj_close': df.get(f'Adj Close_{ticker}'),
            'volume': df.get(f'Volume_{ticker}'),
            'returns': df.get(f'{ticker}_returns'),
            'momentum': df.get(f'{ticker}_momentum'),
            'volatility': df.get(f'{ticker}_volatility'),
            'infl_adj_return': df.get(f'{ticker}_infl_adj_return'),
            'bullish_momentum': df.get(f'{ticker}_bullish_momentum'),
            'vol_unemp_risk': df.get(f'{ticker}_vol_unemp_risk'),
            'gdp': df['gdp'],
            'inflation': df['inflation'],
            'unemployment': df['unemployment']
        })
        long_data.append(ticker_df)
    
    return pd.concat(long_data, ignore_index=True)


def sql_to_wide(db_url: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
    """
    Load data from TimescaleDB market_data table and convert to wide format.
    """

    # db_url: SQLAlchemy database connection string
    # start_date: Optional start date filter (ISO format: 'YYYY-MM-DD')
    # end_date: Optional end date filter (ISO format: 'YYYY-MM-DD')
    # we'll get a wide format DataFrame matching the original Injest output format

    engine = create_engine(db_url)
    
    # Build query with optional date filters
    query = "SELECT * FROM market_data"
    conditions = []
    
    if start_date:
        conditions.append(f"time >= '{start_date}'")
    if end_date:
        conditions.append(f"time <= '{end_date}'")
    
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    
    query += " ORDER BY time, ticker"
    
    # Load data
    df_long = pd.read_sql(query, engine)
    engine.dispose()
    
    # Convert time to datetime
    df_long['time'] = pd.to_datetime(df_long['time'])
    
    wide_dfs = []
    
    # Pivot OHLCV columns (format: Prefix_TICKER)
    ohlcv_cols = ['open', 'high', 'low', 'close', 'adj_close', 'volume']
    for col in ohlcv_cols:
        pivoted = df_long.pivot(index='time', columns='ticker', values=col)
        col_name = 'Adj Close' if col == 'adj_close' else col.title()
        pivoted.columns = [f'{col_name}_{ticker}' for ticker in pivoted.columns]
        wide_dfs.append(pivoted)
    
    # Pivot feature columns (format: TICKER_suffix)
    feature_cols = ['returns', 'momentum', 'volatility', 'infl_adj_return', 'bullish_momentum', 'vol_unemp_risk']
    for col in feature_cols:
        pivoted = df_long.pivot(index='time', columns='ticker', values=col)
        pivoted.columns = [f'{ticker}_{col}' for ticker in pivoted.columns]
        wide_dfs.append(pivoted)
    
    # Economic indicators (same across all tickers)
    econ_df = df_long.groupby('time')[['gdp', 'inflation', 'unemployment']].first()
    wide_dfs.append(econ_df)
    
    # Concatenate all
    df_wide = pd.concat(wide_dfs, axis=1)
    df_wide.index.name = 'Date'
    
    return df_wide

# how I would use the function``: 
# df = sql_to_wide(
#         "postgresql+psycopg2://admin:password@localhost:5432/mytimescale",
#         start_date='2024-01-01',
#         end_date='2024-12-31'
#     )
    
# parameter config for neural forecast LSTM model.
def my_lstm_config(trial: optuna.Trial):
    """Hyperparameter configuration for LSTM"""
    config = {
        'input_size': trial.suggest_int('input_size', 7, 21),
        'encoder_n_layers': trial.suggest_int('encoder_n_layers', 2, 3),
        'encoder_bias': trial.suggest_categorical('encoder_bias', [True, False]),
        'encoder_dropout': trial.suggest_float('encoder_dropout', 0.0, 1.0),
        'learning_rate': trial.suggest_loguniform("learning_rate", 1e-7, 1e-3),
        "encoder_hidden_size": trial.suggest_int("encoder_hidden_size", 32, 256),
        "decoder_hidden_size": trial.suggest_int("decoder_hidden_size", 32, 256),
        'decoder_layers': trial.suggest_int('decoder_layers', 2, 3),
        "batch_size": trial.suggest_int("batch_size", 32, 128),
        "random_seed": trial.suggest_int("random_seed", 1, 10), 
        'early_stop_patience_steps': trial.suggest_int('early_stop_patience_steps', 3, 10),
        "recurrent": trial.suggest_categorical('recurrent', [True, False]),
        'scaler_type': 'robust', 
        "exclude_insample_y": True,
        "optimizer" : trial.suggest_categorical("optimizer", [Adam, RMSprop]),
        "hist_exog_list": ["Close", "High", "Low", "Open", "Volume", "bullish_momentum", "infl_adj_return", "momentum", "returns", "vol_unemp_risk", "volatility", "gdp", "unemployment",	"inflation"],
    }
    trainer_kwargs = {
        'max_steps': trial.suggest_int('max_steps', 100, 300),
        "enable_checkpointing": True,
        'gradient_clip_val': trial.suggest_float('gradient_clip_val', 1.0, 5.0)
    }
    config.update(**trainer_kwargs)
    return config
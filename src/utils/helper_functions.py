import numpy as np
import pandas as pd
from typing import List
from torch.optim import Adam, RMSprop
import optuna


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
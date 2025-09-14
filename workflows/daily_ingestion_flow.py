from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from prefect import flow, task, get_run_logger


@task(name="load_tickers", retries=0)
def load_tickers() -> Dict[str, List[str]]:
    """Return default tickers for stocks and ETFs.

    Adjust this task to load from config or env if desired.
    """
    return {
        "stock_ticker": ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'JNJ', 'PFE', 'JPM', 'GS', 'XOM', 'CVX'],
        "ETF_ticker": ['XLK', 'XLV', 'XLF', 'XLE'],
    }


@task(name="build_injestor")
def build_injestor(
    tickers: Dict[str, List[str]],
    end_date: Optional[dt.datetime] = None,
    up_to_days: int = 6,
):
    """Instantiate the `Injest` class from `src.Data_injestion`.

    Notes:
    - `Data_injestion.py` has a protected `__main__` block to avoid side effects on import.
    - `end_date` defaults to now if not provided.
    """
    from src.Data_injestion import Injest  # lazy import to avoid side-effects during module load

    if end_date is None:
        end_date = dt.datetime.now()
    return Injest(tickers=tickers, end_date=end_date, up_to=up_to_days)


@task(name="aggregate_data")
def aggregate_data(injestor) -> pd.DataFrame:
    """Run the injestor aggregate and return a dataframe."""
    return injestor.aggregate()


@task(name="save_output")
def save_output(df: pd.DataFrame, output_path: str = "data/clean_data/general_data.csv") -> str:
    """Save the aggregated dataframe to CSV and return the saved path."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=True)
    return str(out)


@flow(name="daily-inference-ingestion")
def daily_ingestion_flow(
    tickers: Optional[Dict[str, List[str]]] = None,
    end_date: Optional[str] = None,
    up_to_days: int = 6,
    output_path: str = "data/clean_data/general_data.csv",
):
    """Orchestrate daily ingestion for inference.

    Parameters
    - tickers: Optional override for tickers dict.
    - end_date: Optional ISO date/time string; defaults to now if not provided.
    - up_to_days: Lookback window in days for initial data fetch.
    - output_path: Where to save the aggregated dataset.
    """
    logger = get_run_logger()

    if tickers is None:
        tickers = load_tickers()

    parsed_end: Optional[dt.datetime] = (
        dt.datetime.fromisoformat(end_date) if end_date else None
    )

    injestor = build_injestor(tickers, parsed_end, up_to_days)
    df = aggregate_data(injestor)
    saved = save_output(df, output_path)

    logger.info(f"Saved aggregated data to: {saved}")
    return saved


if __name__ == "__main__":
    # Run once locally (no schedule) for quick testing
    daily_ingestion_flow()


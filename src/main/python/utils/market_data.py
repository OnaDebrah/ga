import time

import yfinance as yf
import numpy as np
import pandas as pd
from typing import Tuple

from utils.logger import logger

def download_price_data(
        tickers: list,
        start: str = "2018-01-01",
        end: str = "2024-12-31",
        retry_count: int = 3,
        pause: float = 1.0
) -> pd.DataFrame:
    """
    Download adjusted close prices with error handling and retries

    Args:
        tickers: List of stock tickers
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
        retry_count: Number of download retries
        pause: Delay between retries in seconds

    Returns:
        DataFrame with adjusted closing prices
    """
    for attempt in range(retry_count):
        try:
            logger.info(f"Downloading data (attempt {attempt + 1}/{retry_count})...")
            data = yf.download(
                tickers,
                start=start,
                end=end,
                #group_by='ticker',
                #progress=True,
                #threads=True
            )["Close"]

            # Handle single ticker case (returns Series instead of DataFrame)
            if isinstance(data, pd.Series):
                data = data.to_frame(name=tickers[0])

            valid_tickers = data.columns.tolist()
            if len(valid_tickers) < len(tickers):
                missing = set(tickers) - set(valid_tickers)
                logger.warning(f"Missing data for: {missing}")

            data = data.dropna(how='all').ffill().bfill()

            if len(data) < 10:  # Sanity check
                raise ValueError("Insufficient data points")

            logger.info(f"Successfully downloaded {len(data)} days of data")
            return data

        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt == retry_count - 1:
                raise
            time.sleep(pause)

def compute_portfolio_statistics(
        price_df: pd.DataFrame,
        min_trading_days: int = 50,
        risk_free_rate: float = 0.02
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Compute portfolio statistics with validation

    Args:
        price_df: DataFrame of adjusted closing prices
        min_trading_days: Minimum required data points
        risk_free_rate: Annual risk-free rate for Sharpe ratio

    Returns:
        Tuple of (mean returns, standard deviations, correlation matrix, risk capacity)
    """
    # Input validation
    if len(price_df) < min_trading_days:
        raise ValueError(f"Need at least {min_trading_days} trading days")

    if price_df.isnull().any().any():
        raise ValueError("Input contains NaN values after cleaning")

    # Calculate daily returns
    returns = price_df.pct_change().dropna()
    n_days = len(returns)

    # Annualization factors
    trading_days_per_year = 252
    annual_factor = np.sqrt(trading_days_per_year)

    # Compute statistics
    mean_returns = returns.mean().values * trading_days_per_year  # Annualized
    std_devs = returns.std().values * annual_factor  # Annualized

    # Handle negative returns for Sharpe ratio
    sharpe_ratios = (mean_returns - risk_free_rate) / (std_devs + 1e-6)

    # Regularized correlation matrix
    corr = returns.corr()
    eigenvalues = np.linalg.eigvals(corr)
    if np.any(eigenvalues <= 0):
        logger.warning("Correlation matrix not positive definite - applying regularization")
        corr = corr + np.eye(len(corr)) * 0.01
        corr = corr / corr.values.diagonal()[:,None]  # Re-normalize

    # Risk capacity (20% of total risk with diversification benefit)
    C = 0.2 * np.sum(std_devs) / np.sqrt(len(std_devs))

    logger.info(f"""
    Computed statistics:
    - Time period: {n_days} trading days
    - Mean returns: {mean_returns.mean():.2%} to {mean_returns.max():.2%} annualized
    - Volatility: {std_devs.mean():.2%} to {std_devs.max():.2%} annualized
    - Risk capacity: {C:.4f}
    """)
    return mean_returns, std_devs, corr.values, float(C)


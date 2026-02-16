"""
Data Loading Utilities for European LSTM Portfolio
Handles WRDS connections, data downloads, and preprocessing
"""

import wrds
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import logging
from getpass import getpass

logger = logging.getLogger(__name__)


class WRDSDataLoader:
    """
    Handles connection to WRDS and downloading European stock data
    """

    def __init__(self, username: Optional[str] = None, password: Optional[str] = None):
        """
        Initialize WRDS connection

        Args:
            username: WRDS username (if None, will prompt)
            password: WRDS password (if None, will prompt securely)
        """
        self.username = username
        self.password = password
        self.db = None

    def connect(self):
        """Establish connection to WRDS"""
        if self.db is not None:
            logger.info("Already connected to WRDS")
            return

        # Prompt for credentials if not provided
        if self.username is None:
            self.username = input("Enter WRDS Username: ")
        if self.password is None:
            self.password = getpass("Enter WRDS Password: ")

        try:
            logger.info("Connecting to WRDS...")
            self.db = wrds.Connection(wrds_username=self.username)
            logger.info("✓ Successfully connected to WRDS")
        except Exception as e:
            logger.error(f"Failed to connect to WRDS: {e}")
            raise

    def disconnect(self):
        """Close WRDS connection"""
        if self.db is not None:
            self.db.close()
            self.db = None
            logger.info("Disconnected from WRDS")

    def get_european_universe(self,
                             country_codes: List[str],
                             start_date: str,
                             end_date: str,
                             min_market_cap: float = 2e9,
                             min_avg_volume: float = 1e6) -> pd.DataFrame:
        """
        Query WRDS Compustat Global for European stocks

        Args:
            country_codes: List of ISO country codes (e.g., ['DEU', 'FRA', 'GBR'])
            start_date: Start date 'YYYY-MM-DD'
            end_date: End date 'YYYY-MM-DD'
            min_market_cap: Minimum market cap in EUR
            min_avg_volume: Minimum average daily volume in EUR

        Returns:
            DataFrame with stock fundamentals and identifiers
        """
        if self.db is None:
            self.connect()

        logger.info(f"Querying European stocks from {start_date} to {end_date}")

        # Build country filter
        country_filter = "', '".join(country_codes)

        # Query Compustat Global Security Daily
        # Note: This is a simplified query - adjust table/columns based on your WRDS access
        query = f"""
        SELECT
            a.gvkey,
            a.iid,
            b.conm as company_name,
            b.fic as country_code,
            b.gsector as gics_sector,
            a.datadate,
            a.prccd as price,
            a.cshoc as shares_outstanding,
            a.cshtrd as volume,
            a.curcdd as currency
        FROM
            comp.g_secd as a
        INNER JOIN
            comp.g_company as b
        ON
            a.gvkey = b.gvkey
        WHERE
            b.fic IN ('{country_filter}')
            AND a.datadate >= '{start_date}'
            AND a.datadate <= '{end_date}'
            AND a.prccd IS NOT NULL
            AND a.cshoc IS NOT NULL
        ORDER BY
            a.gvkey, a.datadate
        """

        try:
            df = self.db.raw_sql(query)
            logger.info(f"✓ Retrieved {len(df)} records from WRDS")

            # Calculate market cap
            df['market_cap'] = df['price'] * df['shares_outstanding']

            # Calculate average daily volume in currency
            df['volume_value'] = df['price'] * df['volume']

            return df

        except Exception as e:
            logger.error(f"Query failed: {e}")
            raise

    def get_fundamentals(self,
                        gvkeys: List[str],
                        start_date: str,
                        end_date: str) -> pd.DataFrame:
        """
        Get fundamental data for specific stocks

        Args:
            gvkeys: List of GVKEY identifiers
            start_date: Start date 'YYYY-MM-DD'
            end_date: End date 'YYYY-MM-DD'

        Returns:
            DataFrame with fundamental ratios
        """
        if self.db is None:
            self.connect()

        gvkey_filter = "', '".join(gvkeys)

        query = f"""
        SELECT
            gvkey,
            datadate,
            fyear,
            revt as revenue,
            ni as net_income,
            at as total_assets,
            ceq as common_equity,
            lt as total_liabilities,
            act as current_assets,
            lct as current_liabilities
        FROM
            comp.g_funda
        WHERE
            gvkey IN ('{gvkey_filter}')
            AND datadate >= '{start_date}'
            AND datadate <= '{end_date}'
            AND indfmt = 'INDL'
            AND datafmt = 'STD'
            AND popsrc = 'D'
            AND consol = 'C'
        ORDER BY
            gvkey, datadate
        """

        try:
            df = self.db.raw_sql(query)
            logger.info(f"✓ Retrieved fundamentals for {len(df)} records")

            # Calculate ratios
            df['roe'] = df['net_income'] / df['common_equity']
            df['roa'] = df['net_income'] / df['total_assets']
            df['debt_to_equity'] = df['total_liabilities'] / df['common_equity']
            df['current_ratio'] = df['current_assets'] / df['current_liabilities']

            return df

        except Exception as e:
            logger.error(f"Fundamentals query failed: {e}")
            raise


class MacroDataLoader:
    """
    Downloads macroeconomic variables using yfinance
    """

    def __init__(self, macro_tickers: Dict[str, str]):
        """
        Initialize with macro variable mapping

        Args:
            macro_tickers: Dict mapping variable names to yfinance tickers
                          e.g., {'BUND_10Y': '^TNX', 'VSTOXX': '^V2TX'}
        """
        self.macro_tickers = macro_tickers

    def download_macro_data(self,
                           start_date: str,
                           end_date: str) -> pd.DataFrame:
        """
        Download all macro variables

        Args:
            start_date: Start date 'YYYY-MM-DD'
            end_date: End date 'YYYY-MM-DD'

        Returns:
            DataFrame with macro variables (index: date, columns: variable names)
        """
        logger.info(f"Downloading macro data from {start_date} to {end_date}")

        macro_data = {}

        for var_name, ticker in self.macro_tickers.items():
            try:
                logger.info(f"  Downloading {var_name} ({ticker})...")
                data = yf.download(ticker, start=start_date, end=end_date, progress=False)

                # Use Close price
                if 'Close' in data.columns:
                    macro_data[var_name] = data['Close']
                else:
                    logger.warning(f"No 'Close' column for {ticker}, skipping")

            except Exception as e:
                logger.warning(f"Failed to download {var_name}: {e}")

        # Combine into single DataFrame
        df = pd.DataFrame(macro_data)
        logger.info(f"✓ Downloaded {len(df.columns)} macro variables, {len(df)} days")

        return df

    def calculate_macro_features(self,
                                macro_df: pd.DataFrame,
                                difference: bool = True,
                                lag_days: int = 1) -> pd.DataFrame:
        """
        Calculate macro features (differences and lags)

        Args:
            macro_df: Raw macro data
            difference: If True, calculate daily differences
            lag_days: Number of days to lag (prevents look-ahead bias)

        Returns:
            Processed macro features
        """
        df = macro_df.copy()

        if difference:
            logger.info("Calculating daily differences for stationarity")
            df = df.diff()

        if lag_days > 0:
            logger.info(f"Lagging macro variables by {lag_days} day(s)")
            df = df.shift(lag_days)

        # Drop NaN rows created by diff() and shift()
        df = df.dropna()

        return df


class BenchmarkDataLoader:
    """
    Downloads benchmark index data
    """

    @staticmethod
    def download_benchmark(ticker: str,
                          start_date: str,
                          end_date: str) -> pd.DataFrame:
        """
        Download benchmark index data

        Args:
            ticker: Benchmark ticker (e.g., '^STOXX')
            start_date: Start date 'YYYY-MM-DD'
            end_date: End date 'YYYY-MM-DD'

        Returns:
            DataFrame with benchmark prices and returns
        """
        logger.info(f"Downloading benchmark: {ticker}")

        try:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)

            # Calculate returns
            data['return'] = data['Close'].pct_change()

            logger.info(f"✓ Downloaded {len(data)} days of benchmark data")
            return data

        except Exception as e:
            logger.error(f"Failed to download benchmark: {e}")
            raise


def filter_liquid_stocks(df: pd.DataFrame,
                        min_market_cap: float,
                        min_avg_volume: float,
                        min_history_days: int = 1260) -> pd.DataFrame:
    """
    Filter stocks by liquidity criteria

    Args:
        df: DataFrame with stock data (must have market_cap, volume_value columns)
        min_market_cap: Minimum market capitalization
        min_avg_volume: Minimum average daily volume
        min_history_days: Minimum number of trading days (5 years ≈ 1260 days)

    Returns:
        Filtered DataFrame
    """
    logger.info("Filtering stocks by liquidity criteria")

    initial_stocks = df['gvkey'].nunique()

    # Group by stock and calculate statistics
    stock_stats = df.groupby('gvkey').agg({
        'market_cap': 'mean',
        'volume_value': 'mean',
        'datadate': 'count'
    }).rename(columns={'datadate': 'num_days'})

    # Apply filters
    mask = (
        (stock_stats['market_cap'] >= min_market_cap) &
        (stock_stats['volume_value'] >= min_avg_volume) &
        (stock_stats['num_days'] >= min_history_days)
    )

    qualified_gvkeys = stock_stats[mask].index.tolist()

    filtered_df = df[df['gvkey'].isin(qualified_gvkeys)]

    final_stocks = len(qualified_gvkeys)
    logger.info(f"✓ Filtered {initial_stocks} → {final_stocks} stocks")

    return filtered_df


def stratified_sector_selection(df: pd.DataFrame,
                                sectors: List[str],
                                stocks_per_sector: int) -> List[str]:
    """
    Select top N stocks from each sector by market cap

    Args:
        df: DataFrame with stock data
        sectors: List of GICS sectors
        stocks_per_sector: Number of stocks to select per sector

    Returns:
        List of selected GVKEYs
    """
    logger.info(f"Selecting {stocks_per_sector} stocks per sector")

    selected_gvkeys = []

    # Get latest data for each stock
    latest_data = df.sort_values('datadate').groupby('gvkey').last().reset_index()

    for sector in sectors:
        sector_stocks = latest_data[latest_data['gics_sector'] == sector]

        if len(sector_stocks) == 0:
            logger.warning(f"No stocks found in sector: {sector}")
            continue

        # Sort by market cap and take top N
        top_stocks = sector_stocks.nlargest(stocks_per_sector, 'market_cap')
        selected_gvkeys.extend(top_stocks['gvkey'].tolist())

        logger.info(f"  {sector}: selected {len(top_stocks)} stocks")

    logger.info(f"✓ Total selected: {len(selected_gvkeys)} stocks")

    return selected_gvkeys


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    print("WRDS Data Loader Module")
    print("This module provides utilities for downloading European stock data")

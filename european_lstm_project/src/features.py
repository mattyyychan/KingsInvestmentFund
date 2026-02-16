"""
Feature Engineering Module for European LSTM Portfolio
Handles PCA, technical indicators, and systematic risk removal
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)


class ResidualReturnCalculator:
    """
    Calculates residual returns (alpha) by removing systematic market risk
    """

    def __init__(self, beta_window: int = 60):
        """
        Args:
            beta_window: Rolling window for beta calculation (default: 60 days)
        """
        self.beta_window = beta_window

    def calculate_rolling_beta(self,
                              stock_returns: pd.Series,
                              market_returns: pd.Series) -> pd.Series:
        """
        Calculate rolling beta using covariance method

        Beta = Cov(Stock, Market) / Var(Market)

        Args:
            stock_returns: Stock return series
            market_returns: Market return series (aligned dates)

        Returns:
            Rolling beta series
        """
        # Align series
        aligned = pd.DataFrame({
            'stock': stock_returns,
            'market': market_returns
        }).dropna()

        # Calculate rolling covariance and variance
        rolling_cov = aligned['stock'].rolling(self.beta_window).cov(aligned['market'])
        rolling_var = aligned['market'].rolling(self.beta_window).var()

        beta = rolling_cov / rolling_var

        return beta

    def calculate_residual_returns(self,
                                   stock_returns: pd.Series,
                                   market_returns: pd.Series,
                                   beta: Optional[pd.Series] = None) -> pd.Series:
        """
        Calculate residual returns: r_residual = r_stock - beta * r_market

        Args:
            stock_returns: Stock return series
            market_returns: Market return series
            beta: Pre-calculated beta (if None, will calculate)

        Returns:
            Residual return series
        """
        if beta is None:
            beta = self.calculate_rolling_beta(stock_returns, market_returns)

        # Align all series
        df = pd.DataFrame({
            'stock': stock_returns,
            'market': market_returns,
            'beta': beta
        }).dropna()

        # Calculate residuals
        residuals = df['stock'] - (df['beta'] * df['market'])

        return residuals


class TechnicalIndicators:
    """
    Calculate technical indicators for feature engineering
    """

    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    @staticmethod
    def calculate_macd(prices: pd.Series,
                      fast: int = 12,
                      slow: int = 26,
                      signal: int = 9) -> Tuple[pd.Series, pd.Series]:
        """
        Moving Average Convergence Divergence

        Returns:
            (macd_line, signal_line)
        """
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()

        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()

        return macd_line, signal_line

    @staticmethod
    def calculate_bollinger_bands(prices: pd.Series,
                                  period: int = 20,
                                  num_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Bollinger Bands

        Returns:
            (upper_band, middle_band, lower_band)
        """
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()

        upper = middle + (num_std * std)
        lower = middle - (num_std * std)

        return upper, middle, lower

    @staticmethod
    def calculate_returns(prices: pd.Series, windows: List[int]) -> pd.DataFrame:
        """
        Calculate returns over multiple windows

        Args:
            prices: Price series
            windows: List of window sizes (e.g., [5, 10, 20, 60])

        Returns:
            DataFrame with columns: return_5d, return_10d, etc.
        """
        returns_df = pd.DataFrame(index=prices.index)

        for window in windows:
            returns_df[f'return_{window}d'] = prices.pct_change(window)

        return returns_df

    @staticmethod
    def calculate_volatility(returns: pd.Series, window: int = 20) -> pd.Series:
        """Rolling volatility (standard deviation of returns)"""
        return returns.rolling(window=window).std()


class FeatureEngineer:
    """
    Main feature engineering class - combines all feature generation
    """

    def __init__(self,
                 beta_window: int = 60,
                 pca_variance_threshold: float = 0.95,
                 fit_on_train_only: bool = True):
        """
        Args:
            beta_window: Window for beta calculation
            pca_variance_threshold: Variance threshold for PCA (default: 0.95)
            fit_on_train_only: If True, fit scalers/PCA only on training data
        """
        self.beta_window = beta_window
        self.pca_variance_threshold = pca_variance_threshold
        self.fit_on_train_only = fit_on_train_only

        # Objects to store fitted transformers
        self.scaler = None
        self.pca = None
        self.technical_feature_names = []

    def create_technical_features(self,
                                  price_data: pd.DataFrame,
                                  volume_data: pd.DataFrame) -> pd.DataFrame:
        """
        Create technical indicator features

        Args:
            price_data: DataFrame with columns [date, ticker, price]
            volume_data: DataFrame with columns [date, ticker, volume]

        Returns:
            DataFrame with technical features
        """
        logger.info("Creating technical indicator features")

        features_list = []

        for ticker in price_data['ticker'].unique():
            ticker_prices = price_data[price_data['ticker'] == ticker].set_index('date')['price']
            ticker_volume = volume_data[volume_data['ticker'] == ticker].set_index('date')['volume']

            # Calculate indicators
            rsi = TechnicalIndicators.calculate_rsi(ticker_prices)
            macd, signal = TechnicalIndicators.calculate_macd(ticker_prices)
            bb_upper, bb_middle, bb_lower = TechnicalIndicators.calculate_bollinger_bands(ticker_prices)

            returns_df = TechnicalIndicators.calculate_returns(ticker_prices, [5, 10, 20, 60])
            volatility = TechnicalIndicators.calculate_volatility(ticker_prices.pct_change())

            # Volume indicators
            volume_ma = ticker_volume.rolling(window=20).mean()

            # Combine into DataFrame
            ticker_features = pd.DataFrame({
                'ticker': ticker,
                'rsi': rsi,
                'macd': macd,
                'macd_signal': signal,
                'bb_upper': bb_upper,
                'bb_middle': bb_middle,
                'bb_lower': bb_lower,
                'volatility': volatility,
                'volume_ma': volume_ma,
            }).join(returns_df)

            features_list.append(ticker_features.reset_index())

        features = pd.concat(features_list, ignore_index=True)
        logger.info(f"✓ Created {len(features.columns) - 2} technical features")

        return features

    def apply_pca(self,
                 technical_features: pd.DataFrame,
                 train_mask: Optional[pd.Series] = None) -> Tuple[pd.DataFrame, PCA]:
        """
        Apply PCA to technical features

        Args:
            technical_features: DataFrame with technical indicators
            train_mask: Boolean mask indicating training data (for fitting)

        Returns:
            (pca_features, fitted_pca_object)
        """
        logger.info("Applying PCA to technical features")

        # Separate feature columns (exclude date, ticker, etc.)
        exclude_cols = ['date', 'ticker', 'gvkey']
        feature_cols = [col for col in technical_features.columns if col not in exclude_cols]

        X = technical_features[feature_cols].copy()

        # Handle missing values
        X = X.fillna(X.mean())

        # Standardize features
        if self.fit_on_train_only and train_mask is not None:
            logger.info("Fitting scaler on training data only")
            self.scaler = StandardScaler()
            X_train = X[train_mask]
            self.scaler.fit(X_train)
            X_scaled = self.scaler.transform(X)
        else:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)

        # Apply PCA
        if self.fit_on_train_only and train_mask is not None:
            logger.info("Fitting PCA on training data only")
            self.pca = PCA(n_components=self.pca_variance_threshold, svd_solver='full')
            self.pca.fit(X_scaled[train_mask])
            X_pca = self.pca.transform(X_scaled)
        else:
            self.pca = PCA(n_components=self.pca_variance_threshold, svd_solver='full')
            X_pca = self.pca.fit_transform(X_scaled)

        n_components = self.pca.n_components_
        explained_var = self.pca.explained_variance_ratio_.sum()

        logger.info(f"✓ PCA: {n_components} components explain {explained_var:.2%} variance")

        # Create DataFrame with PCA components
        pca_cols = [f'pca_{i+1}' for i in range(n_components)]
        pca_df = pd.DataFrame(X_pca, columns=pca_cols, index=technical_features.index)

        # Add back identifiers
        for col in exclude_cols:
            if col in technical_features.columns:
                pca_df[col] = technical_features[col].values

        return pca_df, self.pca


def create_target_variable(residual_returns: pd.Series,
                          threshold: float = 0.0) -> pd.Series:
    """
    Create binary target variable for LSTM training

    Target = 1 if residual_return > threshold, else 0

    Args:
        residual_returns: Residual return series
        threshold: Threshold for classification (default: 0.0)

    Returns:
        Binary target series
    """
    target = (residual_returns > threshold).astype(int)
    return target


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Feature Engineering Module for European LSTM Portfolio")

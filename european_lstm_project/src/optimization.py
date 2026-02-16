"""
Portfolio Optimization Module - Maximum Sharpe Ratio
Uses CVXPY for convex optimization with constraints
"""

import numpy as np
import pandas as pd
import cvxpy as cp
from typing import Dict, List, Optional, Tuple
import logging
from scipy import linalg

logger = logging.getLogger(__name__)


class MaximumSharpeOptimizer:
    """
    Portfolio optimization using Maximum Sharpe Ratio criterion
    """

    def __init__(self,
                 risk_free_rate: float = 0.02,
                 max_position_size: float = 0.10,
                 max_sector_weight: float = 0.15,
                 long_only: bool = True,
                 use_shrinkage: bool = True):
        """
        Initialize optimizer

        Args:
            risk_free_rate: Annual risk-free rate (default: 2%)
            max_position_size: Maximum weight per stock (default: 10%)
            max_sector_weight: Maximum weight per sector (default: 15%)
            long_only: If True, constrain weights >= 0
            use_shrinkage: If True, apply Ledoit-Wolf shrinkage to covariance
        """
        self.risk_free_rate = risk_free_rate
        self.max_position_size = max_position_size
        self.max_sector_weight = max_sector_weight
        self.long_only = long_only
        self.use_shrinkage = use_shrinkage

    def calculate_expected_returns(self,
                                   probabilities: np.ndarray,
                                   avg_up_return: float,
                                   avg_down_return: float) -> np.ndarray:
        """
        Convert LSTM probabilities to expected returns

        E[r] = p * Avg_Up_Return + (1-p) * Avg_Down_Return

        Args:
            probabilities: Array of LSTM probabilities (n_stocks,)
            avg_up_return: Average return when residual is positive
            avg_down_return: Average return when residual is negative

        Returns:
            Expected returns array (n_stocks,)
        """
        expected_returns = (probabilities * avg_up_return +
                          (1 - probabilities) * avg_down_return)

        return expected_returns

    def ledoit_wolf_shrinkage(self, returns: pd.DataFrame) -> np.ndarray:
        """
        Apply Ledoit-Wolf shrinkage to covariance matrix

        Shrinks sample covariance towards constant correlation target

        Args:
            returns: DataFrame of returns (n_samples, n_stocks)

        Returns:
            Shrunk covariance matrix
        """
        from sklearn.covariance import LedoitWolf

        lw = LedoitWolf()
        lw.fit(returns)

        logger.info(f"  Shrinkage intensity: {lw.shrinkage_:.3f}")

        return lw.covariance_

    def optimize_portfolio(self,
                          expected_returns: np.ndarray,
                          covariance_matrix: np.ndarray,
                          sector_map: Optional[Dict[int, str]] = None,
                          tickers: Optional[List[str]] = None) -> Dict:
        """
        Optimize portfolio to maximize Sharpe Ratio

        Reformulation for convexity:
            Original: max (w'μ - rf) / sqrt(w'Σw)
            Reformulated: min y'Σy subject to y'μ = 1, y >= 0
            Then w = y / sum(y)

        Args:
            expected_returns: Expected returns vector (n_stocks,)
            covariance_matrix: Covariance matrix (n_stocks, n_stocks)
            sector_map: Dict mapping stock index to sector name
            tickers: List of stock tickers (for reporting)

        Returns:
            Dict with:
                - weights: Optimal portfolio weights
                - expected_return: Portfolio expected return
                - volatility: Portfolio volatility
                - sharpe_ratio: Sharpe ratio
                - status: Solver status
        """
        n_stocks = len(expected_returns)

        # Decision variable (auxiliary variable y)
        y = cp.Variable(n_stocks)

        # Excess returns
        excess_returns = expected_returns - self.risk_free_rate / 252  # Daily rf

        # Objective: Minimize y' Σ y (variance)
        objective = cp.Minimize(cp.quad_form(y, covariance_matrix))

        # Constraints
        constraints = []

        # Constraint 1: y' μ_excess = 1 (normalization)
        constraints.append(excess_returns @ y == 1)

        # Constraint 2: Long-only (y >= 0)
        if self.long_only:
            constraints.append(y >= 0)

        # Solve problem
        problem = cp.Problem(objective, constraints)

        try:
            problem.solve(solver=cp.ECOS, verbose=False)

            if problem.status != 'optimal':
                logger.warning(f"Optimization status: {problem.status}")
                return self._fallback_equal_weight(n_stocks, tickers)

            # Convert y to weights
            y_opt = y.value
            weights_raw = y_opt / np.sum(y_opt)

            # Apply position size constraints (post-processing)
            weights_constrained = self._apply_position_constraints(
                weights_raw,
                sector_map
            )

            # Calculate portfolio metrics
            portfolio_return = weights_constrained @ expected_returns
            portfolio_variance = weights_constrained @ covariance_matrix @ weights_constrained
            portfolio_vol = np.sqrt(portfolio_variance)
            sharpe_ratio = (portfolio_return - self.risk_free_rate / 252) / portfolio_vol

            result = {
                'weights': weights_constrained,
                'expected_return': portfolio_return,
                'volatility': portfolio_vol,
                'sharpe_ratio': sharpe_ratio,
                'status': 'optimal'
            }

            if tickers:
                result['weights_dict'] = dict(zip(tickers, weights_constrained))

            logger.info(f"✓ Optimization successful - Sharpe: {sharpe_ratio:.3f}")

            return result

        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            return self._fallback_equal_weight(n_stocks, tickers)

    def _apply_position_constraints(self,
                                   weights: np.ndarray,
                                   sector_map: Optional[Dict[int, str]] = None) -> np.ndarray:
        """
        Apply position size and sector constraints via iterative scaling

        Args:
            weights: Raw optimal weights
            sector_map: Mapping of stock index to sector

        Returns:
            Constrained weights (sum = 1)
        """
        weights = weights.copy()

        # Apply max position size
        weights = np.minimum(weights, self.max_position_size)

        # Apply sector constraints if provided
        if sector_map is not None:
            sectors = list(set(sector_map.values()))

            for sector in sectors:
                # Get indices for this sector
                sector_indices = [i for i, s in sector_map.items() if s == sector]
                sector_weight = weights[sector_indices].sum()

                # Scale down if sector weight exceeds limit
                if sector_weight > self.max_sector_weight:
                    scale_factor = self.max_sector_weight / sector_weight
                    weights[sector_indices] *= scale_factor

        # Renormalize to sum to 1
        weights = weights / weights.sum()

        return weights

    def _fallback_equal_weight(self,
                               n_stocks: int,
                               tickers: Optional[List[str]] = None) -> Dict:
        """
        Fallback to equal-weight portfolio if optimization fails

        Args:
            n_stocks: Number of stocks
            tickers: List of tickers

        Returns:
            Equal-weight portfolio dict
        """
        logger.warning("Using equal-weight fallback")

        weights = np.ones(n_stocks) / n_stocks

        result = {
            'weights': weights,
            'expected_return': np.nan,
            'volatility': np.nan,
            'sharpe_ratio': np.nan,
            'status': 'fallback_equal_weight'
        }

        if tickers:
            result['weights_dict'] = dict(zip(tickers, weights))

        return result


class MinimumVarianceOptimizer:
    """
    Minimum Variance portfolio (benchmark strategy)
    """

    @staticmethod
    def optimize(covariance_matrix: np.ndarray,
                tickers: Optional[List[str]] = None) -> Dict:
        """
        Optimize for minimum variance

        Args:
            covariance_matrix: Covariance matrix
            tickers: List of tickers

        Returns:
            Portfolio dict with weights
        """
        n_stocks = covariance_matrix.shape[0]

        # Decision variable
        w = cp.Variable(n_stocks)

        # Objective: Minimize variance
        objective = cp.Minimize(cp.quad_form(w, covariance_matrix))

        # Constraints
        constraints = [
            cp.sum(w) == 1,  # Full investment
            w >= 0           # Long-only
        ]

        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.ECOS, verbose=False)

        weights = w.value

        result = {
            'weights': weights,
            'status': problem.status
        }

        if tickers:
            result['weights_dict'] = dict(zip(tickers, weights))

        return result


def calculate_portfolio_turnover(weights_t: np.ndarray,
                                 weights_t_minus_1: np.ndarray) -> float:
    """
    Calculate portfolio turnover

    Turnover = sum(|w_t - w_t-1|) / 2

    Args:
        weights_t: Current period weights
        weights_t_minus_1: Previous period weights

    Returns:
        Turnover (as a fraction)
    """
    turnover = np.sum(np.abs(weights_t - weights_t_minus_1)) / 2
    return turnover


def apply_transaction_costs(returns: pd.Series,
                           turnover: float,
                           cost_bps: float = 10) -> pd.Series:
    """
    Apply transaction costs to returns

    Args:
        returns: Gross returns
        turnover: Portfolio turnover
        cost_bps: Transaction cost in basis points

    Returns:
        Net returns after costs
    """
    cost = turnover * (cost_bps / 10000)
    net_returns = returns - cost

    return net_returns


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Portfolio Optimization Module - Maximum Sharpe Ratio")

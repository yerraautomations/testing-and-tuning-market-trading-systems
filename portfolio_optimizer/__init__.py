"""
Portfolio Optimizer

A desktop application for optimizing trading strategy portfolios.
Parses MT5 backtest reports and finds optimal strategy combinations
based on correlation and performance metrics.
"""

from .mt5_parser import MT5Parser, Strategy, StrategyMetrics, Trade
from .correlation import CorrelationEngine, CorrelationResult
from .optimizer import (
    PortfolioOptimizer,
    OptimizationConfig,
    CandidateScore,
    PortfolioMetrics
)

__version__ = "0.1.0"
__all__ = [
    "MT5Parser",
    "Strategy",
    "StrategyMetrics",
    "Trade",
    "CorrelationEngine",
    "CorrelationResult",
    "PortfolioOptimizer",
    "OptimizationConfig",
    "CandidateScore",
    "PortfolioMetrics",
]

"""
Correlation Engine

Calculates correlations between trading strategies based on their
daily/weekly returns. Used to find diversified portfolio combinations.

Supports multicore parallel processing for large datasets.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Import from local parser
from mt5_parser import Strategy


@dataclass
class CorrelationResult:
    """Result of correlation calculation between two strategies."""
    strategy_a: str
    strategy_b: str
    correlation: float
    overlap_days: int  # Number of days with data from both strategies


class CorrelationEngine:
    """
    Calculates correlations between trading strategies.

    Correlation is calculated using daily returns aligned by date.
    Only days where both strategies have trades are considered.
    """

    def __init__(self, strategies: List[Strategy]):
        self.strategies = {s.name: s for s in strategies}
        self._daily_returns_cache: Dict[str, Dict[str, float]] = {}
        self._correlation_matrix: Optional[np.ndarray] = None
        self._strategy_names: List[str] = []

    def get_daily_returns(self, strategy: Strategy) -> Dict[str, float]:
        """Get daily returns for a strategy, using cache if available."""
        if strategy.name not in self._daily_returns_cache:
            self._daily_returns_cache[strategy.name] = strategy.get_daily_returns()
        return self._daily_returns_cache[strategy.name]

    def calculate_correlation(
        self,
        strategy_a: Strategy,
        strategy_b: Strategy,
        min_overlap_days: int = 10
    ) -> CorrelationResult:
        """
        Calculate Pearson correlation between two strategies.

        Args:
            strategy_a: First strategy
            strategy_b: Second strategy
            min_overlap_days: Minimum overlapping days required

        Returns:
            CorrelationResult with correlation coefficient and overlap info
        """
        returns_a = self.get_daily_returns(strategy_a)
        returns_b = self.get_daily_returns(strategy_b)

        # Find overlapping dates
        common_dates = set(returns_a.keys()) & set(returns_b.keys())
        overlap_days = len(common_dates)

        if overlap_days < min_overlap_days:
            return CorrelationResult(
                strategy_a=strategy_a.name,
                strategy_b=strategy_b.name,
                correlation=0.0,  # Not enough data
                overlap_days=overlap_days
            )

        # Extract aligned returns
        sorted_dates = sorted(common_dates)
        values_a = [returns_a[d] for d in sorted_dates]
        values_b = [returns_b[d] for d in sorted_dates]

        # Calculate Pearson correlation
        correlation = self._pearson_correlation(values_a, values_b)

        return CorrelationResult(
            strategy_a=strategy_a.name,
            strategy_b=strategy_b.name,
            correlation=correlation,
            overlap_days=overlap_days
        )

    def _pearson_correlation(self, x: List[float], y: List[float]) -> float:
        """Calculate Pearson correlation coefficient."""
        n = len(x)
        if n == 0:
            return 0.0

        mean_x = sum(x) / n
        mean_y = sum(y) / n

        # Calculate covariance and standard deviations
        covariance = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
        std_x = (sum((xi - mean_x) ** 2 for xi in x)) ** 0.5
        std_y = (sum((yi - mean_y) ** 2 for yi in y)) ** 0.5

        if std_x == 0 or std_y == 0:
            return 0.0

        return covariance / (std_x * std_y)

    def build_correlation_matrix(
        self,
        strategies: Optional[List[Strategy]] = None,
        min_overlap_days: int = 10,
        parallel: bool = True,
        max_workers: Optional[int] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Build a correlation matrix for all strategies.

        Args:
            strategies: List of strategies (uses all if None)
            min_overlap_days: Minimum overlapping days required
            parallel: Use parallel processing (default True)
            max_workers: Number of parallel workers (default: CPU count)

        Returns:
            Tuple of (correlation matrix, list of strategy names)
        """
        if strategies is None:
            strategies = list(self.strategies.values())

        n = len(strategies)
        self._strategy_names = [s.name for s in strategies]
        self._correlation_matrix = np.zeros((n, n))

        # Set diagonal to 1.0
        np.fill_diagonal(self._correlation_matrix, 1.0)

        # Calculate number of pairs
        num_pairs = n * (n - 1) // 2

        # Use parallel processing for large matrices
        if parallel and num_pairs > 10:
            self._build_matrix_parallel(strategies, min_overlap_days, max_workers)
        else:
            self._build_matrix_sequential(strategies, min_overlap_days)

        return self._correlation_matrix, self._strategy_names

    def _build_matrix_sequential(
        self,
        strategies: List[Strategy],
        min_overlap_days: int
    ) -> None:
        """Build correlation matrix sequentially."""
        n = len(strategies)
        for i in range(n):
            for j in range(i + 1, n):
                result = self.calculate_correlation(
                    strategies[i],
                    strategies[j],
                    min_overlap_days
                )
                self._correlation_matrix[i, j] = result.correlation
                self._correlation_matrix[j, i] = result.correlation

    def _build_matrix_parallel(
        self,
        strategies: List[Strategy],
        min_overlap_days: int,
        max_workers: Optional[int] = None
    ) -> None:
        """Build correlation matrix using parallel processing."""
        n = len(strategies)
        if max_workers is None:
            max_workers = min(multiprocessing.cpu_count(), n * (n - 1) // 2, 61)  # 61 = Windows pool cap

        # Pre-compute daily returns for all strategies
        daily_returns_list = [s.get_daily_returns() for s in strategies]
        strategy_names = [s.name for s in strategies]

        # Create list of pairs to compute
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append((i, j, daily_returns_list[i], daily_returns_list[j], min_overlap_days))

        # Process pairs in parallel
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_pair = {
                executor.submit(_calculate_correlation_standalone, pair): pair
                for pair in pairs
            }

            for future in as_completed(future_to_pair):
                pair = future_to_pair[future]
                i, j = pair[0], pair[1]
                try:
                    correlation = future.result()
                    self._correlation_matrix[i, j] = correlation
                    self._correlation_matrix[j, i] = correlation
                except Exception as e:
                    print(f"Warning: Could not calculate correlation for pair ({i}, {j}): {e}")

    def get_correlation(self, strategy_a_name: str, strategy_b_name: str) -> float:
        """Get correlation between two strategies by name."""
        if self._correlation_matrix is None:
            self.build_correlation_matrix()

        try:
            i = self._strategy_names.index(strategy_a_name)
            j = self._strategy_names.index(strategy_b_name)
            return self._correlation_matrix[i, j]
        except ValueError:
            return 0.0

    def get_average_correlation_with_group(
        self,
        strategy: Strategy,
        group: List[Strategy]
    ) -> float:
        """
        Calculate average correlation of a strategy with a group.

        Args:
            strategy: The strategy to evaluate
            group: List of strategies to compare against

        Returns:
            Average correlation coefficient
        """
        if not group:
            return 0.0

        correlations = []
        for other in group:
            if other.name != strategy.name:
                result = self.calculate_correlation(strategy, other)
                correlations.append(result.correlation)

        return sum(correlations) / len(correlations) if correlations else 0.0

    def get_max_correlation_with_group(
        self,
        strategy: Strategy,
        group: List[Strategy]
    ) -> float:
        """
        Calculate maximum correlation of a strategy with any in the group.

        Args:
            strategy: The strategy to evaluate
            group: List of strategies to compare against

        Returns:
            Maximum correlation coefficient (absolute value)
        """
        if not group:
            return 0.0

        max_corr = 0.0
        for other in group:
            if other.name != strategy.name:
                result = self.calculate_correlation(strategy, other)
                max_corr = max(max_corr, abs(result.correlation))

        return max_corr

    def to_dict(self) -> Dict:
        """Export correlation matrix as dictionary for JSON serialization."""
        if self._correlation_matrix is None:
            self.build_correlation_matrix()

        return {
            "strategies": self._strategy_names,
            "matrix": self._correlation_matrix.tolist()
        }


def _calculate_correlation_standalone(pair: Tuple) -> float:
    """
    Standalone function for parallel correlation calculation.
    This must be a top-level function for ProcessPoolExecutor to pickle it.

    Args:
        pair: Tuple of (i, j, returns_a, returns_b, min_overlap_days)

    Returns:
        Pearson correlation coefficient
    """
    i, j, returns_a, returns_b, min_overlap_days = pair

    # Find overlapping dates
    common_dates = set(returns_a.keys()) & set(returns_b.keys())
    overlap_days = len(common_dates)

    if overlap_days < min_overlap_days:
        return 0.0

    # Extract aligned returns
    sorted_dates = sorted(common_dates)
    values_a = [returns_a[d] for d in sorted_dates]
    values_b = [returns_b[d] for d in sorted_dates]

    # Calculate Pearson correlation
    n = len(values_a)
    if n == 0:
        return 0.0

    mean_a = sum(values_a) / n
    mean_b = sum(values_b) / n

    covariance = sum((values_a[k] - mean_a) * (values_b[k] - mean_b) for k in range(n))
    std_a = (sum((x - mean_a) ** 2 for x in values_a)) ** 0.5
    std_b = (sum((y - mean_b) ** 2 for y in values_b)) ** 0.5

    if std_a == 0 or std_b == 0:
        return 0.0

    return covariance / (std_a * std_b)


def print_correlation_matrix(matrix: np.ndarray, names: List[str]) -> None:
    """Print correlation matrix in a readable format."""
    # Header
    max_name_len = max(len(n) for n in names)
    header = " " * (max_name_len + 2)
    for name in names:
        header += f"{name[:8]:>10}"
    print(header)
    print("-" * len(header))

    # Rows
    for i, name in enumerate(names):
        row = f"{name:<{max_name_len}}  "
        for j in range(len(names)):
            corr = matrix[i, j]
            row += f"{corr:>10.2f}"
        print(row)


if __name__ == "__main__":
    # Test with sample data
    from mt5_parser import MT5Parser
    import sys

    if len(sys.argv) < 2:
        print("Usage: python correlation.py <directory_with_html_files>")
        sys.exit(1)

    parser = MT5Parser()
    strategies = parser.parse_directory(sys.argv[1])

    if len(strategies) < 2:
        print("Need at least 2 strategies to calculate correlation")
        sys.exit(1)

    print(f"\nLoaded {len(strategies)} strategies")

    engine = CorrelationEngine(strategies)
    matrix, names = engine.build_correlation_matrix()

    print("\nCorrelation Matrix:")
    print_correlation_matrix(matrix, names)

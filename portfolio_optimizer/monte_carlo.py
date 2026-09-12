"""
Monte Carlo Portfolio Analysis

Performs Monte Carlo simulations on portfolios to assess risk and
estimate confidence intervals for key metrics like drawdown and returns.

Supports multicore parallel processing for fast simulation.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import random

from mt5_parser import Strategy


@dataclass
class MonteCarloResult:
    """Results from Monte Carlo simulation."""
    num_simulations: int

    # Final equity statistics
    final_equity_mean: float = 0.0
    final_equity_median: float = 0.0
    final_equity_std: float = 0.0
    final_equity_min: float = 0.0
    final_equity_max: float = 0.0
    final_equity_percentiles: Dict[int, float] = field(default_factory=dict)

    # Max drawdown statistics
    max_dd_mean: float = 0.0
    max_dd_median: float = 0.0
    max_dd_std: float = 0.0
    max_dd_percentiles: Dict[int, float] = field(default_factory=dict)

    # Probability of ruin (hitting certain loss threshold)
    prob_ruin_10pct: float = 0.0  # Probability of 10%+ DD
    prob_ruin_20pct: float = 0.0  # Probability of 20%+ DD
    prob_ruin_30pct: float = 0.0  # Probability of 30%+ DD
    prob_ruin_50pct: float = 0.0  # Probability of 50%+ DD

    # Confidence intervals
    dd_95_ci: float = 0.0  # 95% confidence - max DD won't exceed this
    dd_99_ci: float = 0.0  # 99% confidence - max DD won't exceed this

    # Equity curves from simulations (sample for visualization)
    sample_equity_curves: List[List[float]] = field(default_factory=list)
    percentile_curves: Dict[int, List[float]] = field(default_factory=dict)


@dataclass
class SimulationConfig:
    """Configuration for Monte Carlo simulation."""
    num_simulations: int = 1000
    initial_balance: float = 10000.0

    # Simulation method
    method: str = "shuffle"  # "shuffle" or "bootstrap"

    # For bootstrap resampling
    bootstrap_block_size: int = 5  # Days to keep together

    # Parallel processing
    parallel: bool = True
    max_workers: Optional[int] = None

    # Output options
    num_sample_curves: int = 100  # Number of curves to store for visualization
    percentiles_to_calculate: List[int] = field(
        default_factory=lambda: [5, 10, 25, 50, 75, 90, 95, 99]
    )


class MonteCarloSimulator:
    """
    Monte Carlo simulator for portfolio analysis.

    Supports two simulation methods:
    1. Shuffle: Randomly reorder trades to see alternative equity curves
    2. Bootstrap: Sample trades with replacement for statistical analysis
    """

    def __init__(self, config: Optional[SimulationConfig] = None):
        self.config = config or SimulationConfig()
        self.trades: List[float] = []
        self.daily_returns: Dict[str, float] = {}

    def set_portfolio_trades(self, strategies: List[Strategy]) -> None:
        """
        Extract and combine trades from all strategies in the portfolio.

        Args:
            strategies: List of Strategy objects in the portfolio
        """
        self.trades = []
        self.daily_returns = {}

        for strategy in strategies:
            # Extract closing trade profits
            for trade in strategy.trades:
                if trade.direction == "out" and trade.profit != 0:
                    self.trades.append(trade.profit)

                    # Also track daily returns
                    date_key = trade.time.strftime("%Y-%m-%d")
                    if date_key not in self.daily_returns:
                        self.daily_returns[date_key] = 0.0
                    self.daily_returns[date_key] += trade.profit

    def set_trades_direct(self, trades: List[float]) -> None:
        """Set trades directly from a list of profit values."""
        self.trades = trades

    def set_daily_returns_direct(self, daily_returns: Dict[str, float]) -> None:
        """Set daily returns directly."""
        self.daily_returns = daily_returns

    def run_simulation(self) -> MonteCarloResult:
        """
        Run Monte Carlo simulation with configured parameters.

        Returns:
            MonteCarloResult with statistics and sample curves
        """
        if not self.trades and not self.daily_returns:
            raise ValueError("No trades or daily returns set. Call set_portfolio_trades first.")

        # Use daily returns if available, otherwise use individual trades
        data = list(self.daily_returns.values()) if self.daily_returns else self.trades

        if self.config.parallel and self.config.num_simulations > 100:
            return self._run_parallel(data)
        else:
            return self._run_sequential(data)

    def _run_sequential(self, data: List[float]) -> MonteCarloResult:
        """Run simulations sequentially."""
        results = []

        for i in range(self.config.num_simulations):
            sim_result = _run_single_simulation(
                data,
                self.config.initial_balance,
                self.config.method,
                self.config.bootstrap_block_size,
                i  # seed
            )
            results.append(sim_result)

        return self._aggregate_results(results)

    def _run_parallel(self, data: List[float]) -> MonteCarloResult:
        """Run simulations in parallel using multiple CPU cores with batching."""
        max_workers = self.config.max_workers
        if max_workers is None:
            max_workers = multiprocessing.cpu_count()
        max_workers = min(max_workers, 61)  # Windows process pools are capped at 61 workers

        # Batch simulations for better parallelism
        # Each worker processes multiple simulations to amortize overhead
        sims_per_worker = max(100, self.config.num_simulations // max_workers)
        num_batches = (self.config.num_simulations + sims_per_worker - 1) // sims_per_worker

        batch_args = []
        remaining = self.config.num_simulations
        seed_offset = 0

        for _ in range(num_batches):
            batch_size = min(sims_per_worker, remaining)
            if batch_size <= 0:
                break
            batch_args.append((
                data,
                self.config.initial_balance,
                self.config.method,
                self.config.bootstrap_block_size,
                batch_size,
                seed_offset
            ))
            seed_offset += batch_size
            remaining -= batch_size

        results = []

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit batched simulations
            futures = [
                executor.submit(_run_simulation_batch, *args)
                for args in batch_args
            ]

            # Collect results
            for future in as_completed(futures):
                try:
                    batch_results = future.result()
                    results.extend(batch_results)
                except Exception as e:
                    print(f"Warning: Batch simulation failed: {e}")

        return self._aggregate_results(results)

    def _aggregate_results(
        self,
        sim_results: List[Tuple[List[float], float, float]]
    ) -> MonteCarloResult:
        """
        Aggregate individual simulation results into summary statistics.

        Args:
            sim_results: List of (equity_curve, final_equity, max_drawdown) tuples
        """
        result = MonteCarloResult(num_simulations=len(sim_results))

        if not sim_results:
            return result

        equity_curves = [r[0] for r in sim_results]
        final_equities = [r[1] for r in sim_results]
        max_drawdowns = [r[2] for r in sim_results]

        # Final equity statistics
        result.final_equity_mean = np.mean(final_equities)
        result.final_equity_median = np.median(final_equities)
        result.final_equity_std = np.std(final_equities)
        result.final_equity_min = np.min(final_equities)
        result.final_equity_max = np.max(final_equities)

        # Final equity percentiles
        for p in self.config.percentiles_to_calculate:
            result.final_equity_percentiles[p] = np.percentile(final_equities, p)

        # Max drawdown statistics
        result.max_dd_mean = np.mean(max_drawdowns)
        result.max_dd_median = np.median(max_drawdowns)
        result.max_dd_std = np.std(max_drawdowns)

        # Max DD percentiles
        for p in self.config.percentiles_to_calculate:
            result.max_dd_percentiles[p] = np.percentile(max_drawdowns, p)

        # Probability of ruin calculations
        result.prob_ruin_10pct = sum(1 for dd in max_drawdowns if dd >= 10) / len(max_drawdowns)
        result.prob_ruin_20pct = sum(1 for dd in max_drawdowns if dd >= 20) / len(max_drawdowns)
        result.prob_ruin_30pct = sum(1 for dd in max_drawdowns if dd >= 30) / len(max_drawdowns)
        result.prob_ruin_50pct = sum(1 for dd in max_drawdowns if dd >= 50) / len(max_drawdowns)

        # Confidence intervals for drawdown
        result.dd_95_ci = np.percentile(max_drawdowns, 95)
        result.dd_99_ci = np.percentile(max_drawdowns, 99)

        # Store sample equity curves for visualization
        sample_indices = np.random.choice(
            len(equity_curves),
            min(self.config.num_sample_curves, len(equity_curves)),
            replace=False
        )
        result.sample_equity_curves = [equity_curves[i] for i in sample_indices]

        # Calculate percentile curves (for fan chart)
        if equity_curves:
            curve_length = len(equity_curves[0])
            for p in [5, 25, 50, 75, 95]:
                percentile_curve = []
                for i in range(curve_length):
                    values_at_i = [curve[i] for curve in equity_curves if i < len(curve)]
                    if values_at_i:
                        percentile_curve.append(np.percentile(values_at_i, p))
                result.percentile_curves[p] = percentile_curve

        return result

    def to_dict(self) -> Dict:
        """Export configuration as dictionary for JSON serialization."""
        return {
            "num_simulations": self.config.num_simulations,
            "initial_balance": self.config.initial_balance,
            "method": self.config.method,
            "bootstrap_block_size": self.config.bootstrap_block_size,
            "num_trades": len(self.trades),
            "num_daily_returns": len(self.daily_returns)
        }


def _run_simulation_batch(
    data: List[float],
    initial_balance: float,
    method: str,
    block_size: int,
    batch_size: int,
    seed_offset: int
) -> List[Tuple[List[float], float, float]]:
    """
    Run a batch of Monte Carlo simulations.

    This function runs multiple simulations in a single process call
    to amortize the process creation overhead.

    Args:
        data: List of returns (daily or per-trade)
        initial_balance: Starting balance
        method: "shuffle" or "bootstrap"
        block_size: Block size for bootstrap resampling
        batch_size: Number of simulations to run in this batch
        seed_offset: Starting seed for reproducibility

    Returns:
        List of (equity_curve, final_equity, max_drawdown_pct) tuples
    """
    results = []
    for i in range(batch_size):
        result = _run_single_simulation(
            data, initial_balance, method, block_size, seed_offset + i
        )
        results.append(result)
    return results


def _run_single_simulation(
    data: List[float],
    initial_balance: float,
    method: str,
    block_size: int,
    seed: int
) -> Tuple[List[float], float, float]:
    """
    Run a single Monte Carlo simulation.

    This is a standalone function for parallel processing.

    Args:
        data: List of returns (daily or per-trade)
        initial_balance: Starting balance
        method: "shuffle" or "bootstrap"
        block_size: Block size for bootstrap resampling
        seed: Random seed for reproducibility

    Returns:
        Tuple of (equity_curve, final_equity, max_drawdown_pct)
    """
    rng = random.Random(seed)

    # Generate simulated sequence
    if method == "shuffle":
        # Simple shuffle - reorder returns
        simulated = data.copy()
        rng.shuffle(simulated)
    else:
        # Bootstrap - sample with replacement
        n = len(data)
        if block_size > 1:
            # Block bootstrap - keep some temporal structure
            num_blocks = (n + block_size - 1) // block_size
            simulated = []
            for _ in range(num_blocks):
                start = rng.randint(0, max(0, n - block_size))
                simulated.extend(data[start:start + block_size])
            simulated = simulated[:n]  # Trim to original length
        else:
            # Simple bootstrap
            simulated = [rng.choice(data) for _ in range(n)]

    # Build equity curve
    equity_curve = [initial_balance]
    balance = initial_balance
    peak = initial_balance
    max_drawdown = 0.0

    for ret in simulated:
        balance += ret
        equity_curve.append(balance)

        # Track peak and drawdown
        if balance > peak:
            peak = balance

        if peak > 0:
            drawdown = (peak - balance) / peak * 100
            max_drawdown = max(max_drawdown, drawdown)

    return equity_curve, balance, max_drawdown


def run_portfolio_monte_carlo(
    strategies: List[Strategy],
    num_simulations: int = 1000,
    initial_balance: float = 10000.0,
    method: str = "shuffle",
    parallel: bool = True,
    max_workers: Optional[int] = None
) -> MonteCarloResult:
    """
    Convenience function to run Monte Carlo on a portfolio.

    Args:
        strategies: List of Strategy objects
        num_simulations: Number of simulations to run
        initial_balance: Starting balance
        method: "shuffle" or "bootstrap"
        parallel: Use parallel processing
        max_workers: Number of CPU cores to use

    Returns:
        MonteCarloResult with statistics
    """
    config = SimulationConfig(
        num_simulations=num_simulations,
        initial_balance=initial_balance,
        method=method,
        parallel=parallel,
        max_workers=max_workers
    )

    simulator = MonteCarloSimulator(config)
    simulator.set_portfolio_trades(strategies)

    return simulator.run_simulation()


def print_monte_carlo_summary(result: MonteCarloResult) -> None:
    """Print a formatted summary of Monte Carlo results."""
    print(f"\n{'='*60}")
    print(f"MONTE CARLO SIMULATION RESULTS ({result.num_simulations} simulations)")
    print(f"{'='*60}")

    print(f"\n--- Final Equity Statistics ---")
    print(f"Mean:     ${result.final_equity_mean:,.2f}")
    print(f"Median:   ${result.final_equity_median:,.2f}")
    print(f"Std Dev:  ${result.final_equity_std:,.2f}")
    print(f"Min:      ${result.final_equity_min:,.2f}")
    print(f"Max:      ${result.final_equity_max:,.2f}")

    print(f"\n--- Max Drawdown Statistics ---")
    print(f"Mean DD:    {result.max_dd_mean:.2f}%")
    print(f"Median DD:  {result.max_dd_median:.2f}%")
    print(f"95% CI:     {result.dd_95_ci:.2f}% (95% chance DD stays below this)")
    print(f"99% CI:     {result.dd_99_ci:.2f}% (99% chance DD stays below this)")

    print(f"\n--- Probability of Ruin ---")
    print(f"P(DD >= 10%): {result.prob_ruin_10pct*100:.1f}%")
    print(f"P(DD >= 20%): {result.prob_ruin_20pct*100:.1f}%")
    print(f"P(DD >= 30%): {result.prob_ruin_30pct*100:.1f}%")
    print(f"P(DD >= 50%): {result.prob_ruin_50pct*100:.1f}%")

    print(f"\n--- Final Equity Percentiles ---")
    for p, value in sorted(result.final_equity_percentiles.items()):
        print(f"  {p}th percentile: ${value:,.2f}")


if __name__ == "__main__":
    import sys
    import time
    from mt5_parser import MT5Parser

    print("Monte Carlo Portfolio Simulator")
    print("Usage: python monte_carlo.py <directory_with_html_files> [num_simulations]")

    if len(sys.argv) < 2:
        print("\nExample:")
        print("  python monte_carlo.py ./strategies 1000")
        sys.exit(1)

    directory = sys.argv[1]
    num_sims = int(sys.argv[2]) if len(sys.argv) > 2 else 1000

    # Parse strategies
    parser = MT5Parser()
    strategies = parser.parse_directory(directory)

    if not strategies:
        print(f"No strategies found in {directory}")
        sys.exit(1)

    print(f"\nLoaded {len(strategies)} strategies")
    total_trades = sum(len(s.trades) for s in strategies)
    print(f"Total trades: {total_trades}")

    # Run Monte Carlo
    print(f"\nRunning {num_sims} Monte Carlo simulations...")
    print(f"Using {multiprocessing.cpu_count()} CPU cores")

    start_time = time.time()
    result = run_portfolio_monte_carlo(
        strategies,
        num_simulations=num_sims,
        initial_balance=10000.0,
        method="shuffle",
        parallel=True
    )
    elapsed = time.time() - start_time

    print(f"Completed in {elapsed:.2f} seconds")
    print(f"({num_sims / elapsed:.0f} simulations/second)")

    print_monte_carlo_summary(result)

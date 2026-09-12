"""
Portfolio Optimizer

Finds optimal strategies to add to an existing portfolio based on:
- Low correlation with locked strategies (diversification)
- Strong individual performance metrics
- Combined portfolio improvement
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import numpy as np

from mt5_parser import Strategy, StrategyMetrics
from correlation import CorrelationEngine


@dataclass
class CandidateScore:
    """Scoring result for a candidate strategy."""
    strategy: Strategy

    # Correlation metrics
    avg_correlation: float = 0.0      # Average correlation with locked
    max_correlation: float = 0.0      # Max correlation with any locked

    # Performance metrics (from the strategy itself)
    sharpe_ratio: float = 0.0
    profit_factor: float = 0.0
    recovery_factor: float = 0.0
    max_drawdown_pct: float = 0.0
    win_rate: float = 0.0
    total_trades: int = 0
    net_profit: float = 0.0

    # Combined fitness score
    fitness_score: float = 0.0

    # Reason for ranking
    ranking_reason: str = ""


@dataclass
class PortfolioMetrics:
    """Combined metrics for a portfolio of strategies."""
    strategies: List[str] = field(default_factory=list)
    total_net_profit: float = 0.0
    combined_sharpe: float = 0.0
    combined_max_dd: float = 0.0
    avg_correlation: float = 0.0
    total_trades: int = 0

    # Daily equity curve (date -> cumulative profit)
    equity_curve: Dict[str, float] = field(default_factory=dict)


@dataclass
class OptimizationConfig:
    """Configuration for the optimization process."""
    # How many strategies to recommend
    num_to_add: int = 2

    # Weights for fitness calculation (should sum to 1.0)
    weight_correlation: float = 0.35   # Lower correlation = better
    weight_sharpe: float = 0.25        # Higher Sharpe = better
    weight_profit_factor: float = 0.15 # Higher PF = better
    weight_drawdown: float = 0.15      # Lower DD = better
    weight_recovery: float = 0.10      # Higher recovery = better

    # Filters (candidates must pass these)
    min_sharpe: float = 0.0
    min_profit_factor: float = 1.0
    min_trades: int = 30
    max_correlation: float = 0.7       # Max allowed correlation with locked
    max_drawdown_pct: float = 50.0     # Max allowed drawdown %


class PortfolioOptimizer:
    """
    Optimizes portfolio composition by finding the best candidates
    to add to a set of locked (existing) strategies.
    """

    def __init__(self, config: Optional[OptimizationConfig] = None):
        self.config = config or OptimizationConfig()
        self.locked_strategies: List[Strategy] = []
        self.candidate_strategies: List[Strategy] = []
        self.correlation_engine: Optional[CorrelationEngine] = None
        self.scores: List[CandidateScore] = []

    def set_locked_strategies(self, strategies: List[Strategy]) -> None:
        """Set the strategies that must remain in the portfolio."""
        self.locked_strategies = strategies
        for s in self.locked_strategies:
            s.is_locked = True
        self._rebuild_correlation_engine()

    def set_candidate_strategies(self, strategies: List[Strategy]) -> None:
        """Set the candidate strategies to evaluate."""
        self.candidate_strategies = strategies
        self._rebuild_correlation_engine()

    def _rebuild_correlation_engine(self) -> None:
        """Rebuild correlation engine with all strategies."""
        all_strategies = self.locked_strategies + self.candidate_strategies
        if all_strategies:
            self.correlation_engine = CorrelationEngine(all_strategies)
            self.correlation_engine.build_correlation_matrix()

    def _calculate_candidate_score(self, candidate: Strategy) -> CandidateScore:
        """Calculate the fitness score for a candidate strategy."""
        score = CandidateScore(strategy=candidate)

        # Extract performance metrics
        m = candidate.metrics
        score.sharpe_ratio = m.sharpe_ratio
        score.profit_factor = m.profit_factor
        score.recovery_factor = m.recovery_factor
        score.max_drawdown_pct = m.equity_dd_maximal_pct
        score.win_rate = m.win_rate
        score.total_trades = m.total_trades
        score.net_profit = m.total_net_profit

        # Calculate correlation with locked strategies
        if self.locked_strategies and self.correlation_engine:
            score.avg_correlation = self.correlation_engine.get_average_correlation_with_group(
                candidate, self.locked_strategies
            )
            score.max_correlation = self.correlation_engine.get_max_correlation_with_group(
                candidate, self.locked_strategies
            )

        # Calculate fitness score
        score.fitness_score = self._calculate_fitness(score)
        score.ranking_reason = self._generate_ranking_reason(score)

        return score

    def _calculate_fitness(self, score: CandidateScore) -> float:
        """
        Calculate combined fitness score.

        Higher is better. Components are normalized to 0-1 range.
        """
        cfg = self.config

        # Correlation score (inverted - lower correlation is better)
        # Range: -1 to 1 -> normalized to 0 to 1 (where 0 corr = 0.5, -1 corr = 1.0)
        corr_score = (1 - score.avg_correlation) / 2

        # Sharpe score (normalized, assuming typical range 0-5)
        sharpe_score = min(score.sharpe_ratio / 5.0, 1.0) if score.sharpe_ratio > 0 else 0

        # Profit factor score (normalized, assuming typical range 1-5)
        pf_score = min((score.profit_factor - 1) / 4.0, 1.0) if score.profit_factor > 1 else 0

        # Drawdown score (inverted - lower DD is better)
        # Assuming max DD range 0-50%
        dd_score = max(1 - (score.max_drawdown_pct / 50.0), 0)

        # Recovery factor score (normalized, assuming typical range 0-20)
        recovery_score = min(score.recovery_factor / 20.0, 1.0) if score.recovery_factor > 0 else 0

        # Combined weighted score
        fitness = (
            cfg.weight_correlation * corr_score +
            cfg.weight_sharpe * sharpe_score +
            cfg.weight_profit_factor * pf_score +
            cfg.weight_drawdown * dd_score +
            cfg.weight_recovery * recovery_score
        )

        return round(fitness, 4)

    def _generate_ranking_reason(self, score: CandidateScore) -> str:
        """Generate a human-readable reason for the ranking."""
        reasons = []

        # Correlation insight
        if score.avg_correlation < 0:
            reasons.append(f"Negative correlation ({score.avg_correlation:.2f}) provides hedge")
        elif score.avg_correlation < 0.2:
            reasons.append(f"Very low correlation ({score.avg_correlation:.2f}) = good diversification")
        elif score.avg_correlation < 0.4:
            reasons.append(f"Low correlation ({score.avg_correlation:.2f})")

        # Performance insights
        if score.sharpe_ratio >= 2.0:
            reasons.append(f"Excellent Sharpe ({score.sharpe_ratio:.2f})")
        elif score.sharpe_ratio >= 1.5:
            reasons.append(f"Good Sharpe ({score.sharpe_ratio:.2f})")

        if score.profit_factor >= 2.0:
            reasons.append(f"Strong profit factor ({score.profit_factor:.2f})")

        if score.max_drawdown_pct < 10:
            reasons.append(f"Low drawdown ({score.max_drawdown_pct:.1f}%)")

        return "; ".join(reasons) if reasons else "Balanced metrics"

    def _passes_filters(self, score: CandidateScore) -> bool:
        """Check if candidate passes the minimum filter criteria."""
        cfg = self.config

        if score.sharpe_ratio < cfg.min_sharpe:
            return False
        if score.profit_factor < cfg.min_profit_factor:
            return False
        if score.total_trades < cfg.min_trades:
            return False
        if score.max_correlation > cfg.max_correlation:
            return False
        if score.max_drawdown_pct > cfg.max_drawdown_pct:
            return False

        return True

    def optimize(self) -> List[CandidateScore]:
        """
        Run the optimization and return ranked candidates.

        Returns:
            List of CandidateScore objects, sorted by fitness (best first)
        """
        self.scores = []

        for candidate in self.candidate_strategies:
            score = self._calculate_candidate_score(candidate)

            if self._passes_filters(score):
                self.scores.append(score)

        # Sort by fitness score (descending)
        self.scores.sort(key=lambda s: s.fitness_score, reverse=True)

        return self.scores

    def get_top_recommendations(self, n: Optional[int] = None) -> List[CandidateScore]:
        """Get top N recommendations."""
        if not self.scores:
            self.optimize()

        n = n or self.config.num_to_add
        return self.scores[:n]

    def calculate_portfolio_metrics(
        self,
        strategies: List[Strategy]
    ) -> PortfolioMetrics:
        """
        Calculate combined metrics for a portfolio of strategies.

        Args:
            strategies: List of strategies in the portfolio

        Returns:
            PortfolioMetrics with combined statistics
        """
        metrics = PortfolioMetrics()
        metrics.strategies = [s.name for s in strategies]

        if not strategies:
            return metrics

        # Aggregate basic metrics
        metrics.total_net_profit = sum(s.metrics.total_net_profit for s in strategies)
        metrics.total_trades = sum(s.metrics.total_trades for s in strategies)

        # Combine daily returns for equity curve
        all_daily_returns: Dict[str, float] = {}
        for strategy in strategies:
            daily = strategy.get_daily_returns()
            for date, profit in daily.items():
                if date not in all_daily_returns:
                    all_daily_returns[date] = 0.0
                all_daily_returns[date] += profit

        # Build cumulative equity curve
        cumulative = 0.0
        for date in sorted(all_daily_returns.keys()):
            cumulative += all_daily_returns[date]
            metrics.equity_curve[date] = cumulative

        # Calculate combined Sharpe (simplified - from daily returns)
        if all_daily_returns:
            returns = list(all_daily_returns.values())
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            if std_return > 0:
                # Annualized Sharpe (assuming ~252 trading days)
                metrics.combined_sharpe = (mean_return / std_return) * np.sqrt(252)

        # Calculate combined max drawdown
        if metrics.equity_curve:
            equity_values = [metrics.equity_curve[d] for d in sorted(metrics.equity_curve.keys())]
            metrics.combined_max_dd = self._calculate_max_drawdown(equity_values)

        # Calculate average correlation within portfolio
        if len(strategies) > 1 and self.correlation_engine:
            correlations = []
            for i, s1 in enumerate(strategies):
                for s2 in strategies[i+1:]:
                    corr = self.correlation_engine.get_correlation(s1.name, s2.name)
                    correlations.append(corr)
            metrics.avg_correlation = np.mean(correlations) if correlations else 0.0

        return metrics

    def _calculate_max_drawdown(self, equity_curve: List[float]) -> float:
        """Calculate maximum drawdown percentage from equity curve."""
        if not equity_curve:
            return 0.0

        peak = equity_curve[0]
        max_dd = 0.0

        for value in equity_curve:
            if value > peak:
                peak = value

            if peak > 0:
                dd = (peak - value) / peak * 100
                max_dd = max(max_dd, dd)

        return max_dd

    def compare_portfolios(
        self,
        current: List[Strategy],
        proposed_additions: List[Strategy]
    ) -> Dict:
        """
        Compare current portfolio with proposed additions.

        Returns:
            Dictionary with before/after metrics comparison
        """
        current_metrics = self.calculate_portfolio_metrics(current)
        combined = current + proposed_additions
        proposed_metrics = self.calculate_portfolio_metrics(combined)

        return {
            "current": {
                "strategies": current_metrics.strategies,
                "net_profit": current_metrics.total_net_profit,
                "sharpe": round(current_metrics.combined_sharpe, 2),
                "max_dd": round(current_metrics.combined_max_dd, 2),
                "avg_correlation": round(current_metrics.avg_correlation, 2),
                "equity_curve": current_metrics.equity_curve
            },
            "proposed": {
                "strategies": proposed_metrics.strategies,
                "net_profit": proposed_metrics.total_net_profit,
                "sharpe": round(proposed_metrics.combined_sharpe, 2),
                "max_dd": round(proposed_metrics.combined_max_dd, 2),
                "avg_correlation": round(proposed_metrics.avg_correlation, 2),
                "equity_curve": proposed_metrics.equity_curve
            },
            "improvement": {
                "sharpe_change": round(proposed_metrics.combined_sharpe - current_metrics.combined_sharpe, 2),
                "dd_change": round(proposed_metrics.combined_max_dd - current_metrics.combined_max_dd, 2),
                "correlation_change": round(proposed_metrics.avg_correlation - current_metrics.avg_correlation, 2)
            }
        }

    def to_dict(self) -> Dict:
        """Export optimization results as dictionary for JSON serialization."""
        return {
            "config": {
                "num_to_add": self.config.num_to_add,
                "weights": {
                    "correlation": self.config.weight_correlation,
                    "sharpe": self.config.weight_sharpe,
                    "profit_factor": self.config.weight_profit_factor,
                    "drawdown": self.config.weight_drawdown,
                    "recovery": self.config.weight_recovery
                },
                "filters": {
                    "min_sharpe": self.config.min_sharpe,
                    "min_profit_factor": self.config.min_profit_factor,
                    "min_trades": self.config.min_trades,
                    "max_correlation": self.config.max_correlation,
                    "max_drawdown_pct": self.config.max_drawdown_pct
                }
            },
            "locked_strategies": [s.name for s in self.locked_strategies],
            "candidates_evaluated": len(self.candidate_strategies),
            "candidates_passed_filters": len(self.scores),
            "recommendations": [
                {
                    "rank": i + 1,
                    "name": s.strategy.name,
                    "symbol": s.strategy.symbol,
                    "fitness_score": s.fitness_score,
                    "avg_correlation": round(s.avg_correlation, 3),
                    "max_correlation": round(s.max_correlation, 3),
                    "sharpe": s.sharpe_ratio,
                    "profit_factor": s.profit_factor,
                    "max_dd_pct": s.max_drawdown_pct,
                    "net_profit": s.net_profit,
                    "total_trades": s.total_trades,
                    "reason": s.ranking_reason
                }
                for i, s in enumerate(self.scores)
            ]
        }


def print_recommendations(scores: List[CandidateScore], top_n: int = 5) -> None:
    """Print recommendations in a readable format."""
    print(f"\n{'='*80}")
    print(f"TOP {min(top_n, len(scores))} RECOMMENDATIONS")
    print(f"{'='*80}\n")

    for i, score in enumerate(scores[:top_n]):
        print(f"#{i+1}: {score.strategy.name}")
        print(f"    Symbol: {score.strategy.symbol} | Timeframe: {score.strategy.timeframe}")
        print(f"    Fitness Score: {score.fitness_score:.3f}")
        print(f"    Correlation: avg={score.avg_correlation:.2f}, max={score.max_correlation:.2f}")
        print(f"    Sharpe: {score.sharpe_ratio:.2f} | PF: {score.profit_factor:.2f} | MaxDD: {score.max_drawdown_pct:.1f}%")
        print(f"    Net Profit: ${score.net_profit:,.2f} | Trades: {score.total_trades}")
        print(f"    Why: {score.ranking_reason}")
        print()


if __name__ == "__main__":
    import sys
    from mt5_parser import MT5Parser

    print("Portfolio Optimizer - Command Line Test")
    print("Usage: python optimizer.py <locked_dir> <candidates_dir>")

    if len(sys.argv) < 3:
        print("\nExample:")
        print("  python optimizer.py ./locked_strategies ./candidate_strategies")
        sys.exit(1)

    locked_dir = sys.argv[1]
    candidates_dir = sys.argv[2]

    parser = MT5Parser()

    print(f"\nLoading locked strategies from: {locked_dir}")
    locked = parser.parse_directory(locked_dir)
    print(f"  Found {len(locked)} locked strategies")

    print(f"\nLoading candidates from: {candidates_dir}")
    parser2 = MT5Parser()  # Fresh parser
    candidates = parser2.parse_directory(candidates_dir)
    print(f"  Found {len(candidates)} candidates")

    # Run optimization
    config = OptimizationConfig(
        num_to_add=3,
        min_sharpe=0.5,
        max_correlation=0.6
    )

    optimizer = PortfolioOptimizer(config)
    optimizer.set_locked_strategies(locked)
    optimizer.set_candidate_strategies(candidates)

    scores = optimizer.optimize()
    print_recommendations(scores, top_n=5)

    # Compare portfolios
    if scores:
        top_picks = [scores[0].strategy, scores[1].strategy] if len(scores) > 1 else [scores[0].strategy]
        comparison = optimizer.compare_portfolios(locked, top_picks)

        print("\nPORTFOLIO COMPARISON")
        print("-" * 40)
        print(f"Current Portfolio Sharpe:  {comparison['current']['sharpe']}")
        print(f"Proposed Portfolio Sharpe: {comparison['proposed']['sharpe']}")
        print(f"Sharpe Improvement:        {comparison['improvement']['sharpe_change']:+.2f}")

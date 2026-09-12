"""
API Bridge

Provides a JSON API for the desktop frontend to communicate
with the Python optimization engine.

This can be used either as:
1. A local HTTP server (Flask/FastAPI)
2. Direct Python calls from Tauri via PyO3
3. Subprocess communication via JSON
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import asdict

from mt5_parser import MT5Parser, Strategy, print_strategy_summary
from correlation import CorrelationEngine
from optimizer import PortfolioOptimizer, OptimizationConfig


class PortfolioAPI:
    """
    API interface for the portfolio optimizer.
    All methods return JSON-serializable dictionaries.
    """

    def __init__(self):
        self.parser = MT5Parser()
        self.locked_strategies: List[Strategy] = []
        self.candidate_strategies: List[Strategy] = []
        self.optimizer: Optional[PortfolioOptimizer] = None

    def parse_file(self, file_path: str) -> Dict:
        """
        Parse a single MT5 HTML file.

        Returns:
            Strategy data as dictionary
        """
        try:
            strategy = self.parser.parse_file(file_path)
            return {
                "success": True,
                "data": self._strategy_to_dict(strategy)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def parse_files(self, file_paths: List[str]) -> Dict:
        """
        Parse multiple MT5 HTML files.

        Returns:
            List of strategy data
        """
        results = []
        errors = []

        for path in file_paths:
            try:
                strategy = self.parser.parse_file(path)
                results.append(self._strategy_to_dict(strategy))
            except Exception as e:
                errors.append({"file": path, "error": str(e)})

        return {
            "success": len(errors) == 0,
            "data": results,
            "errors": errors
        }

    def set_locked(self, file_paths: List[str]) -> Dict:
        """Set locked strategies from file paths."""
        self.locked_strategies = []
        parser = MT5Parser()

        errors = []
        for path in file_paths:
            try:
                strategy = parser.parse_file(path)
                strategy.is_locked = True
                self.locked_strategies.append(strategy)
            except Exception as e:
                errors.append({"file": path, "error": str(e)})

        return {
            "success": len(errors) == 0,
            "count": len(self.locked_strategies),
            "strategies": [s.name for s in self.locked_strategies],
            "errors": errors
        }

    def set_candidates(self, file_paths: List[str]) -> Dict:
        """Set candidate strategies from file paths."""
        self.candidate_strategies = []
        parser = MT5Parser()

        errors = []
        for path in file_paths:
            try:
                strategy = parser.parse_file(path)
                self.candidate_strategies.append(strategy)
            except Exception as e:
                errors.append({"file": path, "error": str(e)})

        return {
            "success": len(errors) == 0,
            "count": len(self.candidate_strategies),
            "strategies": [s.name for s in self.candidate_strategies],
            "errors": errors
        }

    def get_correlation_matrix(self) -> Dict:
        """
        Calculate correlation matrix for all loaded strategies.

        Returns:
            Correlation matrix data
        """
        all_strategies = self.locked_strategies + self.candidate_strategies

        if len(all_strategies) < 2:
            return {
                "success": False,
                "error": "Need at least 2 strategies for correlation"
            }

        engine = CorrelationEngine(all_strategies)
        matrix, names = engine.build_correlation_matrix()

        return {
            "success": True,
            "data": {
                "strategies": names,
                "matrix": matrix.tolist(),
                "locked_count": len(self.locked_strategies)
            }
        }

    def optimize(self, config: Optional[Dict] = None) -> Dict:
        """
        Run portfolio optimization.

        Args:
            config: Optional configuration overrides

        Returns:
            Optimization results with recommendations
        """
        if not self.locked_strategies:
            return {
                "success": False,
                "error": "No locked strategies set"
            }

        if not self.candidate_strategies:
            return {
                "success": False,
                "error": "No candidate strategies set"
            }

        # Build config
        opt_config = OptimizationConfig()
        if config:
            if "num_to_add" in config:
                opt_config.num_to_add = config["num_to_add"]
            if "min_sharpe" in config:
                opt_config.min_sharpe = config["min_sharpe"]
            if "min_profit_factor" in config:
                opt_config.min_profit_factor = config["min_profit_factor"]
            if "max_correlation" in config:
                opt_config.max_correlation = config["max_correlation"]
            if "max_drawdown_pct" in config:
                opt_config.max_drawdown_pct = config["max_drawdown_pct"]
            if "min_trades" in config:
                opt_config.min_trades = config["min_trades"]

        # Run optimization
        self.optimizer = PortfolioOptimizer(opt_config)
        self.optimizer.set_locked_strategies(self.locked_strategies)
        self.optimizer.set_candidate_strategies(self.candidate_strategies)
        self.optimizer.optimize()

        return {
            "success": True,
            "data": self.optimizer.to_dict()
        }

    def get_portfolio_comparison(self, selected_indices: List[int]) -> Dict:
        """
        Compare current locked portfolio with selected additions.

        Args:
            selected_indices: Indices of recommended candidates to add

        Returns:
            Before/after comparison data
        """
        if not self.optimizer or not self.optimizer.scores:
            return {
                "success": False,
                "error": "Run optimization first"
            }

        # Get selected strategies
        selected = []
        for idx in selected_indices:
            if 0 <= idx < len(self.optimizer.scores):
                selected.append(self.optimizer.scores[idx].strategy)

        if not selected:
            return {
                "success": False,
                "error": "No valid candidates selected"
            }

        comparison = self.optimizer.compare_portfolios(
            self.locked_strategies,
            selected
        )

        return {
            "success": True,
            "data": comparison
        }

    def get_strategy_equity_curve(self, strategy_name: str) -> Dict:
        """
        Get equity curve data for a specific strategy.

        Returns:
            Date -> cumulative profit data
        """
        # Find strategy
        all_strategies = self.locked_strategies + self.candidate_strategies
        strategy = None
        for s in all_strategies:
            if s.name == strategy_name:
                strategy = s
                break

        if not strategy:
            return {
                "success": False,
                "error": f"Strategy not found: {strategy_name}"
            }

        # Build equity curve from trades
        daily_returns = strategy.get_daily_returns()
        equity_curve = {}
        cumulative = strategy.metrics.initial_deposit

        for date in sorted(daily_returns.keys()):
            cumulative += daily_returns[date]
            equity_curve[date] = round(cumulative, 2)

        return {
            "success": True,
            "data": {
                "name": strategy.name,
                "initial_deposit": strategy.metrics.initial_deposit,
                "equity_curve": equity_curve
            }
        }

    def get_all_equity_curves(self) -> Dict:
        """Get equity curves for all loaded strategies."""
        all_strategies = self.locked_strategies + self.candidate_strategies
        curves = {}

        for strategy in all_strategies:
            daily_returns = strategy.get_daily_returns()
            equity_curve = {}
            cumulative = strategy.metrics.initial_deposit or 10000

            for date in sorted(daily_returns.keys()):
                cumulative += daily_returns[date]
                equity_curve[date] = round(cumulative, 2)

            curves[strategy.name] = {
                "is_locked": strategy.is_locked,
                "symbol": strategy.symbol,
                "equity_curve": equity_curve
            }

        return {
            "success": True,
            "data": curves
        }

    def _strategy_to_dict(self, strategy: Strategy) -> Dict:
        """Convert Strategy object to dictionary."""
        m = strategy.metrics
        return {
            "name": strategy.name,
            "symbol": strategy.symbol,
            "timeframe": strategy.timeframe,
            "period_start": strategy.period_start.isoformat() if strategy.period_start else None,
            "period_end": strategy.period_end.isoformat() if strategy.period_end else None,
            "file_path": strategy.file_path,
            "is_locked": strategy.is_locked,
            "metrics": {
                "net_profit": m.total_net_profit,
                "gross_profit": m.gross_profit,
                "gross_loss": m.gross_loss,
                "profit_factor": m.profit_factor,
                "expected_payoff": m.expected_payoff,
                "recovery_factor": m.recovery_factor,
                "sharpe_ratio": m.sharpe_ratio,
                "max_dd_pct": m.equity_dd_maximal_pct,
                "max_dd_value": m.equity_dd_maximal,
                "total_trades": m.total_trades,
                "win_rate": m.win_rate,
                "initial_deposit": m.initial_deposit
            },
            "trade_count": len(strategy.trades)
        }


# Command-line interface for testing
def main():
    """CLI for testing the API."""
    api = PortfolioAPI()

    if len(sys.argv) < 2:
        print("Portfolio Optimizer API")
        print("Usage: python api.py <command> [args]")
        print("\nCommands:")
        print("  parse <file.html>           - Parse a single file")
        print("  test <locked_dir> <cand_dir> - Run full test")
        return

    command = sys.argv[1]

    if command == "parse" and len(sys.argv) >= 3:
        result = api.parse_file(sys.argv[2])
        print(json.dumps(result, indent=2))

    elif command == "test" and len(sys.argv) >= 4:
        locked_dir = sys.argv[2]
        cand_dir = sys.argv[3]

        # Load strategies
        locked_files = list(Path(locked_dir).glob("*.html"))
        cand_files = list(Path(cand_dir).glob("*.html"))

        print(f"Loading {len(locked_files)} locked strategies...")
        result = api.set_locked([str(f) for f in locked_files])
        print(json.dumps(result, indent=2))

        print(f"\nLoading {len(cand_files)} candidates...")
        result = api.set_candidates([str(f) for f in cand_files])
        print(json.dumps(result, indent=2))

        print("\nCalculating correlations...")
        result = api.get_correlation_matrix()
        if result["success"]:
            print(f"Matrix size: {len(result['data']['strategies'])}x{len(result['data']['strategies'])}")

        print("\nRunning optimization...")
        result = api.optimize({"num_to_add": 3, "max_correlation": 0.6})
        print(json.dumps(result, indent=2))

    else:
        print(f"Unknown command: {command}")


if __name__ == "__main__":
    main()

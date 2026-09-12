"""
MT5 HTML Backtest Report Parser

Parses MT5 Strategy Tester HTML reports to extract:
- Strategy metadata (name, symbol, period)
- Performance metrics (Net Profit, Drawdown, Sharpe, etc.)
- Trade data for correlation analysis
"""

import re
import codecs
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing


@dataclass
class Trade:
    """Represents a single deal/trade from the backtest."""
    time: datetime
    deal_id: int
    symbol: str
    trade_type: str  # buy/sell
    direction: str   # in/out
    volume: float
    price: float
    profit: float
    balance: float
    comment: str = ""


@dataclass
class StrategyMetrics:
    """Key performance metrics extracted from the report."""
    total_net_profit: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    profit_factor: float = 0.0
    expected_payoff: float = 0.0
    recovery_factor: float = 0.0
    sharpe_ratio: float = 0.0

    # Drawdown metrics
    balance_dd_absolute: float = 0.0
    balance_dd_maximal: float = 0.0
    balance_dd_maximal_pct: float = 0.0
    equity_dd_absolute: float = 0.0
    equity_dd_maximal: float = 0.0
    equity_dd_maximal_pct: float = 0.0

    # Trade statistics
    total_trades: int = 0
    profit_trades: int = 0
    loss_trades: int = 0
    win_rate: float = 0.0

    # Additional metrics
    z_score: float = 0.0
    lr_correlation: float = 0.0
    initial_deposit: float = 0.0


@dataclass
class Strategy:
    """Complete strategy data parsed from MT5 HTML report."""
    name: str
    symbol: str
    timeframe: str
    period_start: Optional[datetime] = None
    period_end: Optional[datetime] = None
    metrics: StrategyMetrics = field(default_factory=StrategyMetrics)
    trades: List[Trade] = field(default_factory=list)
    file_path: str = ""

    # For portfolio building
    is_locked: bool = False

    def get_daily_returns(self) -> Dict[str, float]:
        """Calculate daily returns from trades."""
        daily_profits = {}
        for trade in self.trades:
            if trade.direction == "out" and trade.profit != 0:
                date_key = trade.time.strftime("%Y-%m-%d")
                if date_key not in daily_profits:
                    daily_profits[date_key] = 0.0
                daily_profits[date_key] += trade.profit
        return daily_profits

    def get_weekly_returns(self) -> Dict[str, float]:
        """Calculate weekly returns from trades."""
        weekly_profits = {}
        for trade in self.trades:
            if trade.direction == "out" and trade.profit != 0:
                # Get ISO week number
                week_key = trade.time.strftime("%Y-W%W")
                if week_key not in weekly_profits:
                    weekly_profits[week_key] = 0.0
                weekly_profits[week_key] += trade.profit
        return weekly_profits


class MT5Parser:
    """Parser for MT5 Strategy Tester HTML reports."""

    def __init__(self):
        self.strategies: List[Strategy] = []

    def parse_file(self, file_path: str) -> Strategy:
        """Parse a single MT5 HTML report file."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Try UTF-16 first (common for MT5), then UTF-8
        content = None
        for encoding in ['utf-16', 'utf-8', 'latin-1']:
            try:
                with codecs.open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                break
            except (UnicodeDecodeError, UnicodeError):
                continue

        if content is None:
            raise ValueError(f"Unable to decode file: {file_path}")

        strategy = self._parse_content(content)
        strategy.file_path = str(path.absolute())
        self.strategies.append(strategy)
        return strategy

    def _parse_content(self, content: str) -> Strategy:
        """Parse the HTML content and extract strategy data."""
        # Extract strategy name (Expert)
        name_match = re.search(r'Expert:</td>\s*<td[^>]*><b>([^<]+)</b>', content, re.IGNORECASE)
        name = name_match.group(1).strip() if name_match else "Unknown"

        # Extract symbol
        symbol_match = re.search(r'Symbol:</td>\s*<td[^>]*><b>([^<]+)</b>', content, re.IGNORECASE)
        symbol = symbol_match.group(1).strip() if symbol_match else "Unknown"

        # Extract period (timeframe and date range)
        period_match = re.search(r'Period:</td>\s*<td[^>]*><b>([^<]+)</b>', content, re.IGNORECASE)
        timeframe = "Unknown"
        period_start = None
        period_end = None

        if period_match:
            period_str = period_match.group(1).strip()
            # Parse "M5 (2024.11.01 - 2025.11.28)"
            tf_match = re.match(r'(\w+)\s*\((\d{4}\.\d{2}\.\d{2})\s*-\s*(\d{4}\.\d{2}\.\d{2})\)', period_str)
            if tf_match:
                timeframe = tf_match.group(1)
                try:
                    period_start = datetime.strptime(tf_match.group(2), "%Y.%m.%d")
                    period_end = datetime.strptime(tf_match.group(3), "%Y.%m.%d")
                except ValueError:
                    pass

        # Create strategy object
        strategy = Strategy(
            name=name,
            symbol=symbol,
            timeframe=timeframe,
            period_start=period_start,
            period_end=period_end
        )

        # Parse metrics
        strategy.metrics = self._parse_metrics(content)

        # Parse trades
        strategy.trades = self._parse_trades(content)

        return strategy

    def _parse_metrics(self, content: str) -> StrategyMetrics:
        """Extract performance metrics from the HTML content."""
        metrics = StrategyMetrics()

        # Helper function to extract numeric value
        def extract_value(pattern: str, default: float = 0.0) -> float:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                value_str = match.group(1).replace(' ', '').replace(',', '')
                try:
                    return float(value_str)
                except ValueError:
                    return default
            return default

        def extract_value_with_pct(pattern: str) -> Tuple[float, float]:
            """Extract value and percentage from patterns like '43.61 (0.42%)'"""
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                value_str = match.group(1).replace(' ', '').replace(',', '')
                try:
                    value = float(value_str)
                    # Try to extract percentage
                    pct_match = re.search(r'\(([\d.]+)%\)', match.group(0))
                    pct = float(pct_match.group(1)) if pct_match else 0.0
                    return value, pct
                except ValueError:
                    return 0.0, 0.0
            return 0.0, 0.0

        # Core metrics
        metrics.total_net_profit = extract_value(r'Total Net Profit:</td>\s*<td[^>]*><b>([\d\s.,\-]+)</b>')
        metrics.gross_profit = extract_value(r'Gross Profit:</td>\s*<td[^>]*><b>([\d\s.,\-]+)</b>')
        metrics.gross_loss = extract_value(r'Gross Loss:</td>\s*<td[^>]*><b>([\d\s.,\-]+)</b>')
        metrics.profit_factor = extract_value(r'Profit Factor:</td>\s*<td[^>]*><b>([\d\s.,\-]+)</b>')
        metrics.expected_payoff = extract_value(r'Expected Payoff:</td>\s*<td[^>]*><b>([\d\s.,\-]+)</b>')
        metrics.recovery_factor = extract_value(r'Recovery Factor:</td>\s*<td[^>]*><b>([\d\s.,\-]+)</b>')
        metrics.sharpe_ratio = extract_value(r'Sharpe Ratio:</td>\s*<td[^>]*><b>([\d\s.,\-]+)</b>')

        # Drawdown metrics
        metrics.balance_dd_absolute = extract_value(r'Balance Drawdown Absolute:</td>\s*<td[^>]*><b>([\d\s.,\-]+)</b>')

        # Balance Drawdown Maximal with percentage
        dd_match = re.search(r'Balance Drawdown Maximal:</td>\s*<td[^>]*><b>([\d\s.,\-]+)\s*\(([\d.]+)%\)</b>', content)
        if dd_match:
            metrics.balance_dd_maximal = float(dd_match.group(1).replace(' ', '').replace(',', ''))
            metrics.balance_dd_maximal_pct = float(dd_match.group(2))

        metrics.equity_dd_absolute = extract_value(r'Equity Drawdown Absolute:</td>\s*<td[^>]*><b>([\d\s.,\-]+)</b>')

        # Equity Drawdown Maximal with percentage
        eq_dd_match = re.search(r'Equity Drawdown Maximal:</td>\s*<td[^>]*><b>([\d\s.,\-]+)\s*\(([\d.]+)%\)</b>', content)
        if eq_dd_match:
            metrics.equity_dd_maximal = float(eq_dd_match.group(1).replace(' ', '').replace(',', ''))
            metrics.equity_dd_maximal_pct = float(eq_dd_match.group(2))

        # Trade statistics
        metrics.total_trades = int(extract_value(r'Total Trades:</td>\s*<td[^>]*><b>(\d+)</b>'))

        profit_trades_match = re.search(r'Profit Trades \(% of total\):</td>\s*<td[^>]*><b>(\d+)\s*\(([\d.]+)%\)</b>', content)
        if profit_trades_match:
            metrics.profit_trades = int(profit_trades_match.group(1))
            metrics.win_rate = float(profit_trades_match.group(2))

        loss_trades_match = re.search(r'Loss Trades \(% of total\):</td>\s*<td[^>]*><b>(\d+)', content)
        if loss_trades_match:
            metrics.loss_trades = int(loss_trades_match.group(1))

        # Additional metrics
        z_score_match = re.search(r'Z-Score:</td>\s*<td[^>]*><b>([\d\s.,\-]+)', content)
        if z_score_match:
            metrics.z_score = float(z_score_match.group(1).replace(' ', '').replace(',', ''))

        metrics.lr_correlation = extract_value(r'LR Correlation:</td>\s*<td[^>]*><b>([\d\s.,\-]+)</b>')
        metrics.initial_deposit = extract_value(r'Initial Deposit:</td>\s*<td[^>]*><b>([\d\s.,]+)</b>')

        return metrics

    def _parse_trades(self, content: str) -> List[Trade]:
        """Extract trades from the Deals table."""
        trades = []

        # Find the Deals section
        deals_match = re.search(r'<b>Deals</b>.*?</table>', content, re.DOTALL)
        if not deals_match:
            return trades

        deals_section = deals_match.group(0)

        # Parse each trade row
        # Pattern: <tr bgcolor="..."><td>time</td><td>deal</td><td>symbol</td><td>type</td><td>direction</td><td>volume</td><td>price</td><td>order</td><td>commission</td><td>swap</td><td>profit</td><td>balance</td><td>comment</td></tr>
        row_pattern = re.compile(
            r'<tr[^>]*>\s*'
            r'<td>(\d{4}\.\d{2}\.\d{2}\s+\d{2}:\d{2}:\d{2})</td>\s*'  # Time
            r'<td>(\d+)</td>\s*'  # Deal ID
            r'<td>([^<]*)</td>\s*'  # Symbol
            r'<td>([^<]*)</td>\s*'  # Type
            r'<td>([^<]*)</td>\s*'  # Direction
            r'<td>([^<]*)</td>\s*'  # Volume
            r'<td>([^<]*)</td>\s*'  # Price
            r'<td>([^<]*)</td>\s*'  # Order
            r'<td>[^<]*</td>\s*'  # Commission (skip)
            r'<td>[^<]*</td>\s*'  # Swap (skip)
            r'<td>([^<]*)</td>\s*'  # Profit
            r'<td>([^<]*)</td>\s*'  # Balance
            r'<td>([^<]*)</td>',  # Comment
            re.IGNORECASE
        )

        for match in row_pattern.finditer(deals_section):
            try:
                time_str = match.group(1)
                deal_id = int(match.group(2))
                symbol = match.group(3).strip()
                trade_type = match.group(4).strip()
                direction = match.group(5).strip()

                # Parse volume (might be empty)
                volume_str = match.group(6).strip().replace(' ', '')
                volume = float(volume_str) if volume_str else 0.0

                # Parse price
                price_str = match.group(7).strip().replace(' ', '')
                price = float(price_str) if price_str else 0.0

                # Parse profit
                profit_str = match.group(9).strip().replace(' ', '').replace(',', '')
                profit = float(profit_str) if profit_str else 0.0

                # Parse balance
                balance_str = match.group(10).strip().replace(' ', '').replace(',', '')
                balance = float(balance_str) if balance_str else 0.0

                comment = match.group(11).strip()

                # Skip balance operations
                if trade_type == "balance":
                    continue

                trade = Trade(
                    time=datetime.strptime(time_str, "%Y.%m.%d %H:%M:%S"),
                    deal_id=deal_id,
                    symbol=symbol,
                    trade_type=trade_type,
                    direction=direction,
                    volume=volume,
                    price=price,
                    profit=profit,
                    balance=balance,
                    comment=comment
                )
                trades.append(trade)

            except (ValueError, IndexError) as e:
                continue  # Skip malformed rows

        return trades

    def parse_directory(
        self,
        directory: str,
        pattern: str = "*.html",
        parallel: bool = True,
        max_workers: Optional[int] = None
    ) -> List[Strategy]:
        """
        Parse all HTML files in a directory.

        Args:
            directory: Path to directory containing HTML files
            pattern: Glob pattern for HTML files
            parallel: Use parallel processing (default True)
            max_workers: Number of parallel workers (default: CPU count)

        Returns:
            List of parsed Strategy objects
        """
        path = Path(directory)
        if not path.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")

        file_paths = list(path.glob(pattern))

        if not file_paths:
            return []

        # Use parallel processing for multiple files
        if parallel and len(file_paths) > 1:
            strategies = self._parse_files_parallel(file_paths, max_workers)
        else:
            strategies = self._parse_files_sequential(file_paths)

        # Deterministic order regardless of parallel completion order, and
        # unique names so same-named reports (e.g. one EA, many parameter
        # sets) are not merged by the correlation engine.
        strategies.sort(key=lambda s: s.file_path)
        make_names_unique(strategies)
        return strategies

    def _parse_files_sequential(self, file_paths: List[Path]) -> List[Strategy]:
        """Parse files sequentially."""
        strategies = []
        for file_path in file_paths:
            try:
                strategy = self.parse_file(str(file_path))
                strategies.append(strategy)
            except Exception as e:
                print(f"Warning: Could not parse {file_path}: {e}")
        return strategies

    def _parse_files_parallel(
        self,
        file_paths: List[Path],
        max_workers: Optional[int] = None
    ) -> List[Strategy]:
        """Parse files in parallel using multiple CPU cores."""
        if max_workers is None:
            max_workers = min(len(file_paths), multiprocessing.cpu_count(), 61)  # 61 = Windows pool cap

        strategies = []
        file_path_strs = [str(fp) for fp in file_paths]

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all parsing tasks
            future_to_path = {
                executor.submit(_parse_file_standalone, fp): fp
                for fp in file_path_strs
            }

            # Collect results as they complete
            for future in as_completed(future_to_path):
                file_path = future_to_path[future]
                try:
                    strategy = future.result()
                    if strategy:
                        strategies.append(strategy)
                        self.strategies.append(strategy)
                except Exception as e:
                    print(f"Warning: Could not parse {file_path}: {e}")

        return strategies


def make_names_unique(strategies: List[Strategy]) -> List[Strategy]:
    """
    Give duplicate strategy names a suffix so every strategy is addressable
    by name. MT5 / StrategyQuant produce many reports with the same Expert
    name (one EA, different parameter sets); without this they collide in
    the correlation engine, portfolio lookups and databanks.

    The suffix is the report file's stem (e.g. 'ReportTester-4000065623'),
    so a name always traces back to its file. Names already unique are
    left untouched. Modifies the strategies in place and returns them.
    """
    counts = {}
    for s in strategies:
        counts[s.name] = counts.get(s.name, 0) + 1
    seen = set()
    for s in strategies:
        if counts[s.name] > 1:
            stem = Path(s.file_path).stem if s.file_path else ''
            candidate = f"{s.name} [{stem}]" if stem else f"{s.name} #{len(seen) + 1}"
            n = 2
            while candidate in seen:
                candidate = f"{s.name} [{stem}] #{n}"
                n += 1
            s.name = candidate
        seen.add(s.name)
    return strategies


def _parse_file_standalone(file_path: str) -> Optional[Strategy]:
    """
    Standalone function for parallel file parsing.
    This must be a top-level function for ProcessPoolExecutor to pickle it.
    """
    parser = MT5Parser()
    try:
        return parser.parse_file(file_path)
    except Exception as e:
        print(f"Warning: Could not parse {file_path}: {e}")
        return None


def print_strategy_summary(strategy: Strategy) -> None:
    """Print a formatted summary of the strategy."""
    print(f"\n{'='*60}")
    print(f"Strategy: {strategy.name}")
    print(f"{'='*60}")
    print(f"Symbol: {strategy.symbol} | Timeframe: {strategy.timeframe}")
    if strategy.period_start and strategy.period_end:
        print(f"Period: {strategy.period_start.strftime('%Y-%m-%d')} to {strategy.period_end.strftime('%Y-%m-%d')}")

    m = strategy.metrics
    print(f"\n--- Performance Metrics ---")
    print(f"Net Profit:      ${m.total_net_profit:,.2f}")
    print(f"Profit Factor:   {m.profit_factor:.2f}")
    print(f"Sharpe Ratio:    {m.sharpe_ratio:.2f}")
    print(f"Recovery Factor: {m.recovery_factor:.2f}")
    print(f"Max Drawdown:    ${m.equity_dd_maximal:,.2f} ({m.equity_dd_maximal_pct:.2f}%)")
    print(f"Total Trades:    {m.total_trades}")
    print(f"Win Rate:        {m.win_rate:.2f}%")
    print(f"Trades Parsed:   {len(strategy.trades)}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python mt5_parser.py <html_file_or_directory>")
        sys.exit(1)

    parser = MT5Parser()
    target = sys.argv[1]

    if Path(target).is_file():
        strategy = parser.parse_file(target)
        print_strategy_summary(strategy)
    elif Path(target).is_dir():
        strategies = parser.parse_directory(target)
        for strategy in strategies:
            print_strategy_summary(strategy)
    else:
        print(f"Error: {target} is not a valid file or directory")
        sys.exit(1)

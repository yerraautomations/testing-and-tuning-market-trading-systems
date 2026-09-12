"""
Portfolio Optimizer - Streamlit App

A visual tool for building and optimizing trading strategy portfolios
from MT5 backtest HTML reports.

Features:
- Upload and parse MT5 HTML files
- View all strategies in a databank
- Filter by DD, Sharpe, Profit Factor, etc.
- Build portfolios with constraints (min/max strategies, max per symbol)
- Lock strategies and find optimal additions
- Real-time optimization with live results
- View portfolio analysis with equity curves and correlation heatmaps
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import json
import os
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from itertools import combinations
import threading
from datetime import datetime

# Import our backend modules
import sys
sys.path.insert(0, str(Path(__file__).parent))

from mt5_parser import MT5Parser, Strategy, StrategyMetrics, Trade, make_names_unique
from correlation import CorrelationEngine
from optimizer import PortfolioOptimizer, OptimizationConfig
from monte_carlo import MonteCarloSimulator, SimulationConfig, run_portfolio_monte_carlo

# Databank folder
DATABANK_FOLDER = Path(__file__).parent / "databanks"

# Page config
st.set_page_config(
    page_title="Portfolio Optimizer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


def apply_theme(dark_mode: bool):
    """Apply CSS theme based on dark/light mode selection."""
    if dark_mode:
        st.markdown("""
        <style>
            .stApp {
                background-color: #0f172a;
                color: #e2e8f0;
            }
            .metric-card {
                background-color: #1e293b;
                border-radius: 8px;
                padding: 1rem;
                border: 1px solid #334155;
            }
            .strategy-row {
                background-color: #1e293b;
                border-radius: 4px;
                padding: 0.5rem;
                margin-bottom: 0.25rem;
            }
            .locked-badge {
                background-color: #22c55e20;
                color: #22c55e;
                padding: 2px 8px;
                border-radius: 4px;
                font-size: 0.75rem;
            }
            .candidate-badge {
                background-color: #3b82f620;
                color: #3b82f6;
                padding: 2px 8px;
                border-radius: 4px;
                font-size: 0.75rem;
            }
        </style>
        """, unsafe_allow_html=True)
    else:
        # Light theme
        st.markdown("""
        <style>
            .stApp {
                background-color: #ffffff;
                color: #1e293b;
            }
            .metric-card {
                background-color: #f8fafc;
                border-radius: 8px;
                padding: 1rem;
                border: 1px solid #e2e8f0;
            }
            .strategy-row {
                background-color: #f8fafc;
                border-radius: 4px;
                padding: 0.5rem;
                margin-bottom: 0.25rem;
            }
            .locked-badge {
                background-color: #22c55e20;
                color: #16a34a;
                padding: 2px 8px;
                border-radius: 4px;
                font-size: 0.75rem;
            }
            .candidate-badge {
                background-color: #3b82f620;
                color: #2563eb;
                padding: 2px 8px;
                border-radius: 4px;
                font-size: 0.75rem;
            }
        </style>
        """, unsafe_allow_html=True)


# ============================================================================
# DATABANK SAVE/LOAD FUNCTIONS
# ============================================================================

def ensure_databank_folder():
    """Create databank folder if it doesn't exist."""
    DATABANK_FOLDER.mkdir(exist_ok=True)


def get_saved_databanks() -> List[str]:
    """Get list of saved databank files."""
    ensure_databank_folder()
    files = list(DATABANK_FOLDER.glob("*.json"))
    return sorted([f.stem for f in files], reverse=True)  # Most recent first by name


def strategy_to_json_dict(strategy: Strategy) -> Dict:
    """Convert a Strategy object to a dictionary for JSON serialization."""
    return {
        'name': strategy.name,
        'symbol': strategy.symbol,
        'timeframe': strategy.timeframe,
        'period_start': strategy.period_start.isoformat() if strategy.period_start else None,
        'period_end': strategy.period_end.isoformat() if strategy.period_end else None,
        'file_path': strategy.file_path,
        'is_locked': strategy.is_locked,
        'metrics': {
            'total_net_profit': strategy.metrics.total_net_profit,
            'gross_profit': strategy.metrics.gross_profit,
            'gross_loss': strategy.metrics.gross_loss,
            'profit_factor': strategy.metrics.profit_factor,
            'expected_payoff': strategy.metrics.expected_payoff,
            'recovery_factor': strategy.metrics.recovery_factor,
            'sharpe_ratio': strategy.metrics.sharpe_ratio,
            'balance_dd_absolute': strategy.metrics.balance_dd_absolute,
            'balance_dd_maximal': strategy.metrics.balance_dd_maximal,
            'balance_dd_maximal_pct': strategy.metrics.balance_dd_maximal_pct,
            'equity_dd_absolute': strategy.metrics.equity_dd_absolute,
            'equity_dd_maximal': strategy.metrics.equity_dd_maximal,
            'equity_dd_maximal_pct': strategy.metrics.equity_dd_maximal_pct,
            'total_trades': strategy.metrics.total_trades,
            'profit_trades': strategy.metrics.profit_trades,
            'loss_trades': strategy.metrics.loss_trades,
            'win_rate': strategy.metrics.win_rate,
            'z_score': strategy.metrics.z_score,
            'lr_correlation': strategy.metrics.lr_correlation,
            'initial_deposit': strategy.metrics.initial_deposit,
        },
        'trades': [
            {
                'time': t.time.isoformat(),
                'deal_id': t.deal_id,
                'symbol': t.symbol,
                'trade_type': t.trade_type,
                'direction': t.direction,
                'volume': t.volume,
                'price': t.price,
                'profit': t.profit,
                'balance': t.balance,
                'comment': t.comment,
            }
            for t in strategy.trades
        ]
    }


def dict_to_strategy(data: Dict) -> Strategy:
    """Convert a dictionary back to a Strategy object."""
    metrics = StrategyMetrics(
        total_net_profit=data['metrics']['total_net_profit'],
        gross_profit=data['metrics']['gross_profit'],
        gross_loss=data['metrics']['gross_loss'],
        profit_factor=data['metrics']['profit_factor'],
        expected_payoff=data['metrics']['expected_payoff'],
        recovery_factor=data['metrics']['recovery_factor'],
        sharpe_ratio=data['metrics']['sharpe_ratio'],
        balance_dd_absolute=data['metrics']['balance_dd_absolute'],
        balance_dd_maximal=data['metrics']['balance_dd_maximal'],
        balance_dd_maximal_pct=data['metrics']['balance_dd_maximal_pct'],
        equity_dd_absolute=data['metrics']['equity_dd_absolute'],
        equity_dd_maximal=data['metrics']['equity_dd_maximal'],
        equity_dd_maximal_pct=data['metrics']['equity_dd_maximal_pct'],
        total_trades=data['metrics']['total_trades'],
        profit_trades=data['metrics']['profit_trades'],
        loss_trades=data['metrics']['loss_trades'],
        win_rate=data['metrics']['win_rate'],
        z_score=data['metrics']['z_score'],
        lr_correlation=data['metrics']['lr_correlation'],
        initial_deposit=data['metrics']['initial_deposit'],
    )

    trades = [
        Trade(
            time=datetime.fromisoformat(t['time']),
            deal_id=t['deal_id'],
            symbol=t['symbol'],
            trade_type=t['trade_type'],
            direction=t['direction'],
            volume=t['volume'],
            price=t['price'],
            profit=t['profit'],
            balance=t['balance'],
            comment=t['comment'],
        )
        for t in data['trades']
    ]

    strategy = Strategy(
        name=data['name'],
        symbol=data['symbol'],
        timeframe=data['timeframe'],
        period_start=datetime.fromisoformat(data['period_start']) if data['period_start'] else None,
        period_end=datetime.fromisoformat(data['period_end']) if data['period_end'] else None,
        metrics=metrics,
        trades=trades,
        file_path=data['file_path'],
        is_locked=data.get('is_locked', False),
    )

    return strategy


def save_databank(name: str, strategies: List[Strategy], locked_indices: set) -> str:
    """Save current databank to a JSON file."""
    ensure_databank_folder()

    # Sanitize filename
    safe_name = "".join(c for c in name if c.isalnum() or c in (' ', '-', '_')).strip()
    if not safe_name:
        safe_name = f"databank_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    filepath = DATABANK_FOLDER / f"{safe_name}.json"

    data = {
        'saved_at': datetime.now().isoformat(),
        'num_strategies': len(strategies),
        'locked_indices': list(locked_indices),
        'strategies': [strategy_to_json_dict(s) for s in strategies]
    }

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

    return str(filepath)


def load_databank(name: str) -> Tuple[List[Strategy], set]:
    """Load a databank from a JSON file."""
    filepath = DATABANK_FOLDER / f"{name}.json"

    if not filepath.exists():
        raise FileNotFoundError(f"Databank '{name}' not found")

    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    strategies = [dict_to_strategy(s) for s in data['strategies']]
    locked_indices = set(data.get('locked_indices', []))

    return strategies, locked_indices


def delete_databank(name: str) -> bool:
    """Delete a saved databank."""
    filepath = DATABANK_FOLDER / f"{name}.json"
    if filepath.exists():
        filepath.unlink()
        return True
    return False


# ============================================================================
# PORTFOLIO SAVE/LOAD FUNCTIONS
# ============================================================================

PORTFOLIO_FOLDER = Path(__file__).parent / "saved_portfolios"


def ensure_portfolio_folder():
    """Create portfolio folder if it doesn't exist."""
    PORTFOLIO_FOLDER.mkdir(exist_ok=True)


def get_saved_portfolios() -> List[str]:
    """Get list of saved portfolio files."""
    ensure_portfolio_folder()
    files = list(PORTFOLIO_FOLDER.glob("*.json"))
    return sorted([f.stem for f in files], reverse=True)


def portfolio_to_dict(portfolio: 'Portfolio') -> Dict:
    """Convert a Portfolio object to a dictionary for JSON serialization."""
    return {
        'strategy_indices': portfolio.strategy_indices,
        'strategy_names': portfolio.strategy_names,
        'total_profit': portfolio.total_profit,
        'balance_dd': portfolio.balance_dd,
        'equity_dd': portfolio.equity_dd,
        'return_dd_ratio': portfolio.return_dd_ratio,
        'avg_correlation': portfolio.avg_correlation,
        'sharpe': portfolio.sharpe,
        'symbols': portfolio.symbols,
        'stability': portfolio.stability,
        'symmetry': portfolio.symmetry,
    }


def dict_to_portfolio(data: Dict) -> 'Portfolio':
    """Convert a dictionary back to a Portfolio object."""
    return Portfolio(
        strategy_indices=data['strategy_indices'],
        strategy_names=data['strategy_names'],
        total_profit=data['total_profit'],
        balance_dd=data['balance_dd'],
        equity_dd=data['equity_dd'],
        return_dd_ratio=data['return_dd_ratio'],
        avg_correlation=data['avg_correlation'],
        sharpe=data['sharpe'],
        symbols=data['symbols'],
        stability=data.get('stability', 0.0),
        symmetry=data.get('symmetry', 0.0),
    )


def save_portfolios(name: str, portfolios: List['Portfolio']) -> str:
    """Save portfolios to a JSON file."""
    ensure_portfolio_folder()

    # Sanitize filename
    safe_name = "".join(c for c in name if c.isalnum() or c in (' ', '-', '_')).strip()
    if not safe_name:
        safe_name = f"portfolios_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    filepath = PORTFOLIO_FOLDER / f"{safe_name}.json"

    data = {
        'saved_at': datetime.now().isoformat(),
        'num_portfolios': len(portfolios),
        'portfolios': [portfolio_to_dict(p) for p in portfolios]
    }

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

    return str(filepath)


def load_portfolios(name: str) -> List['Portfolio']:
    """Load portfolios from a JSON file."""
    filepath = PORTFOLIO_FOLDER / f"{name}.json"

    if not filepath.exists():
        raise FileNotFoundError(f"Portfolio file '{name}' not found")

    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return [dict_to_portfolio(p) for p in data['portfolios']]


def delete_saved_portfolios(name: str) -> bool:
    """Delete a saved portfolio file."""
    filepath = PORTFOLIO_FOLDER / f"{name}.json"
    if filepath.exists():
        filepath.unlink()
        return True
    return False


# ============================================================================
# EXPORT FUNCTIONS - Excel and HTML Reports
# ============================================================================

import io

def build_equity_curve(strategies: List[Strategy]) -> pd.DataFrame:
    """Build combined equity curve from multiple strategies."""
    all_trades = []
    for i, strat in enumerate(strategies):
        for trade in strat.trades:
            # Closed trades only ("out" / "in/out" deals). Entry deals carry
            # no P/L and would distort stability, symmetry and trade counts.
            if trade.direction.lower() in ("out", "in/out"):
                all_trades.append({
                    'datetime': trade.time,
                    'profit': trade.profit,
                    'strategy': strat.name
                })

    if not all_trades:
        return pd.DataFrame()

    df = pd.DataFrame(all_trades)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    df['cumulative_profit'] = df['profit'].cumsum()
    df['balance'] = 10000 + df['cumulative_profit']  # Starting balance assumption
    return df


def calculate_drawdown_series(equity_series: pd.Series) -> pd.Series:
    """Calculate drawdown series from equity curve."""
    rolling_max = equity_series.expanding().max()
    drawdown = equity_series - rolling_max
    return drawdown


# ============================================================================
# ADVANCED STATISTICS CALCULATIONS
# ============================================================================

def calculate_advanced_stats(strategies: List[Strategy], initial_capital: float = 10000) -> Dict:
    """Calculate advanced portfolio statistics like CAGR, SQN, Z-Score, etc."""

    # Build equity curve from trades with actual profit/loss
    all_trades = []
    for strat in strategies:
        for trade in strat.trades:
            # Closed trades only ("out" / "in/out" deals), not every deal row.
            if trade.direction.lower() in ("out", "in/out"):
                all_trades.append({
                    'datetime': trade.time,
                    'profit': trade.profit,
                    'type': trade.trade_type,  # Buy/Sell
                    'strategy': strat.name
                })

    if not all_trades:
        return {}

    df = pd.DataFrame(all_trades)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    df['cumulative_profit'] = df['profit'].cumsum()
    df['balance'] = initial_capital + df['cumulative_profit']

    # Basic stats
    total_trades = len(df)
    total_profit = df['profit'].sum()
    final_balance = initial_capital + total_profit

    wins = df[df['profit'] > 0]
    losses = df[df['profit'] <= 0]
    num_wins = len(wins)
    num_losses = len(losses)

    gross_profit = wins['profit'].sum() if len(wins) > 0 else 0
    gross_loss = abs(losses['profit'].sum()) if len(losses) > 0 else 0

    avg_win = wins['profit'].mean() if len(wins) > 0 else 0
    avg_loss = abs(losses['profit'].mean()) if len(losses) > 0 else 0

    largest_win = wins['profit'].max() if len(wins) > 0 else 0
    largest_loss = losses['profit'].min() if len(losses) > 0 else 0

    # Win rate
    win_rate = num_wins / total_trades if total_trades > 0 else 0

    # Profit Factor
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Average trade
    avg_trade = df['profit'].mean()

    # Standard deviation of trades
    std_trade = df['profit'].std() if len(df) > 1 else 0

    # CAGR (Compound Annual Growth Rate)
    if len(df) > 1:
        start_date = df['datetime'].min()
        end_date = df['datetime'].max()
        years = (end_date - start_date).days / 365.25
        if years > 0 and initial_capital > 0 and final_balance > 0:
            cagr = (final_balance / initial_capital) ** (1 / years) - 1
        else:
            cagr = 0
    else:
        cagr = 0

    # SQN (System Quality Number) = sqrt(N) * Expectancy / StdDev
    if std_trade > 0 and total_trades > 0:
        expectancy = avg_trade
        sqn = (expectancy / std_trade) * np.sqrt(total_trades)
    else:
        sqn = 0

    # R-Expectancy (if we assume R = avg_loss as risk unit)
    if avg_loss > 0:
        r_expectancy = avg_trade / avg_loss
    else:
        r_expectancy = 0

    # Z-Score (measures trade dependency)
    # Z = (N*(R-0.5)-P) / sqrt(P*(P-N)/(N-1))
    # Where N = total trades, R = runs, P = 2*W*L
    if total_trades > 2:
        # Count runs (consecutive wins or losses)
        runs = 1
        for i in range(1, len(df)):
            if (df.iloc[i]['profit'] > 0) != (df.iloc[i-1]['profit'] > 0):
                runs += 1

        P = 2 * num_wins * num_losses
        if P > 0 and (P - total_trades) != 0:
            z_score = (total_trades * (runs - 0.5) - P) / np.sqrt((P * (P - total_trades)) / (total_trades - 1))
        else:
            z_score = 0
    else:
        z_score = 0

    # Consecutive wins/losses
    max_consec_wins = 0
    max_consec_losses = 0
    current_wins = 0
    current_losses = 0

    for profit in df['profit']:
        if profit > 0:
            current_wins += 1
            current_losses = 0
            max_consec_wins = max(max_consec_wins, current_wins)
        else:
            current_losses += 1
            current_wins = 0
            max_consec_losses = max(max_consec_losses, current_losses)

    # Stagnation (longest flat period)
    if len(df) > 1:
        peak_balance = initial_capital
        peak_date = df['datetime'].min()
        max_stagnation_days = 0

        for _, row in df.iterrows():
            if row['balance'] > peak_balance:
                # New peak - calculate stagnation from last peak
                stagnation = (row['datetime'] - peak_date).days
                max_stagnation_days = max(max_stagnation_days, stagnation)
                peak_balance = row['balance']
                peak_date = row['datetime']

        # Check stagnation from last peak to end
        final_stagnation = (df['datetime'].max() - peak_date).days
        max_stagnation_days = max(max_stagnation_days, final_stagnation)

        total_days = (df['datetime'].max() - df['datetime'].min()).days
        stagnation_pct = max_stagnation_days / total_days if total_days > 0 else 0
    else:
        max_stagnation_days = 0
        stagnation_pct = 0

    # Stability (R² of equity curve)
    if len(df) > 2:
        from scipy import stats
        x = np.arange(len(df))
        y = df['balance'].values
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        stability = r_value ** 2
    else:
        stability = 0

    # Symmetry (ratio of profit consistency)
    # Measures how evenly distributed profits are
    if len(df) > 1:
        # Split trades into halves and compare
        mid = len(df) // 2
        first_half_profit = df.iloc[:mid]['profit'].sum()
        second_half_profit = df.iloc[mid:]['profit'].sum()
        total = abs(first_half_profit) + abs(second_half_profit)
        if total > 0:
            symmetry = 1 - abs(first_half_profit - second_half_profit) / total
        else:
            symmetry = 0
    else:
        symmetry = 0

    # Long/Short breakdown
    long_trades = df[df['type'].str.lower().str.contains('buy', na=False)]
    short_trades = df[df['type'].str.lower().str.contains('sell', na=False)]

    long_profit = long_trades['profit'].sum() if len(long_trades) > 0 else 0
    short_profit = short_trades['profit'].sum() if len(short_trades) > 0 else 0

    # P/L by hour
    df['hour'] = df['datetime'].dt.hour
    pl_by_hour = df.groupby('hour')['profit'].sum().to_dict()

    # P/L by day of week
    df['dayofweek'] = df['datetime'].dt.dayofweek
    pl_by_day = df.groupby('dayofweek')['profit'].sum().to_dict()
    day_names = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}
    pl_by_day_named = {day_names.get(k, k): v for k, v in pl_by_day.items()}

    return {
        'total_trades': total_trades,
        'total_profit': total_profit,
        'num_wins': num_wins,
        'num_losses': num_losses,
        'win_rate': win_rate,
        'gross_profit': gross_profit,
        'gross_loss': gross_loss,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'largest_win': largest_win,
        'largest_loss': largest_loss,
        'avg_trade': avg_trade,
        'profit_factor': profit_factor,
        'cagr': cagr,
        'sqn': sqn,
        'r_expectancy': r_expectancy,
        'z_score': z_score,
        'max_consec_wins': max_consec_wins,
        'max_consec_losses': max_consec_losses,
        'stagnation_days': max_stagnation_days,
        'stagnation_pct': stagnation_pct,
        'stability': stability,
        'symmetry': symmetry,
        'long_trades': len(long_trades),
        'short_trades': len(short_trades),
        'long_profit': long_profit,
        'short_profit': short_profit,
        'pl_by_hour': pl_by_hour,
        'pl_by_day': pl_by_day_named,
        'trades_df': df  # Return for detailed trade list
    }


def calculate_portfolio_stability(strategies: List[Strategy]) -> float:
    """Calculate stability (R²) of portfolio equity curve."""
    equity_df = build_equity_curve(strategies)
    if len(equity_df) < 3:
        return 0.0

    from scipy import stats
    x = np.arange(len(equity_df))
    y = equity_df['balance'].values
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    return r_value ** 2


def calculate_portfolio_symmetry(strategies: List[Strategy]) -> float:
    """Calculate symmetry of portfolio returns."""
    equity_df = build_equity_curve(strategies)
    if len(equity_df) < 2:
        return 0.0

    mid = len(equity_df) // 2
    first_half = equity_df.iloc[:mid]['profit'].sum()
    second_half = equity_df.iloc[mid:]['profit'].sum()
    total = abs(first_half) + abs(second_half)

    if total > 0:
        return 1 - abs(first_half - second_half) / total
    return 0.0


def generate_excel_report(
    portfolios: List['Portfolio'],
    strategies: List[Strategy],
    selected_idx: int = 0,
    monte_carlo_result: dict = None
) -> bytes:
    """Generate Excel report with embedded charts."""
    import xlsxwriter

    output = io.BytesIO()
    workbook = xlsxwriter.Workbook(output, {'in_memory': True})

    # Formats
    header_format = workbook.add_format({'bold': True, 'bg_color': '#4472C4', 'font_color': 'white', 'border': 1})
    number_format = workbook.add_format({'num_format': '#,##0.00', 'border': 1})
    currency_format = workbook.add_format({'num_format': '$#,##0.00', 'border': 1})
    percent_format = workbook.add_format({'num_format': '0.00%', 'border': 1})
    cell_format = workbook.add_format({'border': 1})

    # ==================== Sheet 1: Portfolio Rankings ====================
    ws_rankings = workbook.add_worksheet('Portfolio Rankings')

    headers = ['Rank', 'Strategies', 'Net Profit', 'Equity DD', 'Balance DD',
               'Return/DD', 'Avg Correlation', 'Sharpe', 'Num Strategies']
    for col, header in enumerate(headers):
        ws_rankings.write(0, col, header, header_format)

    for row, p in enumerate(portfolios[:100], start=1):  # Top 100
        ws_rankings.write(row, 0, row, cell_format)
        ws_rankings.write(row, 1, ', '.join(p.strategy_names[:3]) + ('...' if len(p.strategy_names) > 3 else ''), cell_format)
        ws_rankings.write(row, 2, p.total_profit, currency_format)
        ws_rankings.write(row, 3, p.equity_dd, currency_format)
        ws_rankings.write(row, 4, p.balance_dd, currency_format)
        ws_rankings.write(row, 5, p.return_dd_ratio, number_format)
        ws_rankings.write(row, 6, p.avg_correlation, number_format)
        ws_rankings.write(row, 7, p.sharpe, number_format)
        ws_rankings.write(row, 8, len(p.strategy_names), cell_format)

    ws_rankings.set_column('A:A', 6)
    ws_rankings.set_column('B:B', 40)
    ws_rankings.set_column('C:E', 12)
    ws_rankings.set_column('F:H', 12)

    # ==================== Sheet 2: Selected Portfolio Details ====================
    if portfolios and selected_idx < len(portfolios):
        selected = portfolios[selected_idx]
        ws_detail = workbook.add_worksheet('Selected Portfolio')

        # Summary section
        ws_detail.write(0, 0, 'Selected Portfolio Analysis', header_format)
        ws_detail.merge_range('A1:D1', 'Selected Portfolio Analysis', header_format)

        summary_data = [
            ('Total Net Profit', f'${selected.total_profit:,.2f}'),
            ('Equity Drawdown', f'${selected.equity_dd:,.2f}'),
            ('Balance Drawdown', f'${selected.balance_dd:,.2f}'),
            ('Return/DD Ratio', f'{selected.return_dd_ratio:.2f}'),
            ('Average Correlation', f'{selected.avg_correlation:.3f}'),
            ('Sharpe Ratio', f'{selected.sharpe:.2f}'),
            ('Number of Strategies', str(len(selected.strategy_names))),
        ]

        for row, (label, value) in enumerate(summary_data, start=2):
            ws_detail.write(row, 0, label, cell_format)
            ws_detail.write(row, 1, value, cell_format)

        # Strategy list
        ws_detail.write(10, 0, 'Strategies in Portfolio', header_format)
        for row, name in enumerate(selected.strategy_names, start=11):
            ws_detail.write(row, 0, name, cell_format)

        # Build equity curve for selected portfolio
        selected_strategies = [strategies[i] for i in selected.strategy_indices if i < len(strategies)]
        equity_df = build_equity_curve(selected_strategies)

        if not equity_df.empty:
            # Write equity data
            ws_equity = workbook.add_worksheet('Equity Curve Data')
            ws_equity.write(0, 0, 'Date', header_format)
            ws_equity.write(0, 1, 'Balance', header_format)
            ws_equity.write(0, 2, 'Drawdown', header_format)

            dd_series = calculate_drawdown_series(equity_df['balance'])

            for row, (idx, data) in enumerate(equity_df.iterrows(), start=1):
                ws_equity.write(row, 0, str(data['datetime'].date()), cell_format)
                ws_equity.write(row, 1, data['balance'], currency_format)
                ws_equity.write(row, 2, dd_series.iloc[idx], currency_format)

            # Create equity curve chart
            chart = workbook.add_chart({'type': 'line'})
            chart.add_series({
                'name': 'Balance',
                'categories': f'=\'Equity Curve Data\'!$A$2:$A${len(equity_df)+1}',
                'values': f'=\'Equity Curve Data\'!$B$2:$B${len(equity_df)+1}',
                'line': {'color': '#4472C4', 'width': 1.5},
            })
            chart.set_title({'name': 'Portfolio Equity Curve'})
            chart.set_x_axis({'name': 'Date', 'date_axis': True})
            chart.set_y_axis({'name': 'Balance ($)'})
            chart.set_size({'width': 720, 'height': 400})
            ws_detail.insert_chart('E2', chart)

            # Create drawdown chart
            dd_chart = workbook.add_chart({'type': 'area'})
            dd_chart.add_series({
                'name': 'Drawdown',
                'categories': f'=\'Equity Curve Data\'!$A$2:$A${len(equity_df)+1}',
                'values': f'=\'Equity Curve Data\'!$C$2:$C${len(equity_df)+1}',
                'fill': {'color': '#FF6B6B'},
                'line': {'color': '#FF6B6B'},
            })
            dd_chart.set_title({'name': 'Portfolio Drawdown'})
            dd_chart.set_x_axis({'name': 'Date'})
            dd_chart.set_y_axis({'name': 'Drawdown ($)'})
            dd_chart.set_size({'width': 720, 'height': 300})
            ws_detail.insert_chart('E22', dd_chart)

    # ==================== Sheet 3: All Strategies ====================
    ws_strategies = workbook.add_worksheet('All Strategies')
    strat_headers = ['Name', 'Symbol', 'Net Profit', 'Max DD', 'Trades', 'Win Rate',
                     'Profit Factor', 'Sharpe', 'Avg Trade']
    for col, header in enumerate(strat_headers):
        ws_strategies.write(0, col, header, header_format)

    for row, strat in enumerate(strategies, start=1):
        ws_strategies.write(row, 0, strat.name, cell_format)
        ws_strategies.write(row, 1, strat.symbol, cell_format)
        ws_strategies.write(row, 2, strat.metrics.total_net_profit, currency_format)
        ws_strategies.write(row, 3, strat.metrics.equity_dd_maximal, currency_format)
        ws_strategies.write(row, 4, strat.metrics.total_trades, cell_format)
        ws_strategies.write(row, 5, strat.metrics.win_rate / 100, percent_format)
        ws_strategies.write(row, 6, strat.metrics.profit_factor, number_format)
        ws_strategies.write(row, 7, strat.metrics.sharpe_ratio, number_format)
        ws_strategies.write(row, 8, strat.metrics.expected_payoff, currency_format)

    ws_strategies.set_column('A:A', 30)
    ws_strategies.set_column('B:B', 12)
    ws_strategies.set_column('C:I', 12)

    # ==================== Sheet 4: Monte Carlo (if available) ====================
    if monte_carlo_result:
        ws_mc = workbook.add_worksheet('Monte Carlo')

        ws_mc.write(0, 0, 'Monte Carlo Simulation Results', header_format)
        ws_mc.merge_range('A1:D1', 'Monte Carlo Simulation Results', header_format)

        mc_summary = [
            ('Simulations', monte_carlo_result.get('num_simulations', 'N/A')),
            ('Mean Final Balance', f"${monte_carlo_result.get('mean_final_balance', 0):,.2f}"),
            ('Median Final Balance', f"${monte_carlo_result.get('median_final_balance', 0):,.2f}"),
            ('5th Percentile', f"${monte_carlo_result.get('percentile_5', 0):,.2f}"),
            ('95th Percentile', f"${monte_carlo_result.get('percentile_95', 0):,.2f}"),
            ('Max Drawdown (Mean)', f"${monte_carlo_result.get('mean_max_dd', 0):,.2f}"),
            ('Max Drawdown (95th)', f"${monte_carlo_result.get('dd_percentile_95', 0):,.2f}"),
            ('Risk of Ruin (50%)', f"{monte_carlo_result.get('risk_of_ruin_50', 0):.1%}"),
        ]

        for row, (label, value) in enumerate(mc_summary, start=2):
            ws_mc.write(row, 0, label, cell_format)
            ws_mc.write(row, 1, str(value), cell_format)

    workbook.close()
    output.seek(0)
    return output.getvalue()


def generate_html_report(
    portfolios: List['Portfolio'],
    strategies: List[Strategy],
    selected_idx: int = 0,
    monte_carlo_result: dict = None
) -> str:
    """Generate interactive HTML report mirroring the Analysis tab."""

    html_parts = []

    # HTML Header with styling
    html_parts.append("""
<!DOCTYPE html>
<html>
<head>
    <title>Portfolio Analysis Report</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0; padding: 20px; background: #f5f5f5;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        .card {
            background: white; border-radius: 8px; padding: 20px;
            margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        h1 { color: #1a1a2e; border-bottom: 3px solid #4472C4; padding-bottom: 10px; }
        h2 { color: #4472C4; margin-top: 0; }
        h3 { color: #333; margin-top: 20px; }
        table { border-collapse: collapse; width: 100%; margin: 10px 0; }
        th { background: #4472C4; color: white; padding: 12px; text-align: left; }
        td { padding: 10px; border-bottom: 1px solid #ddd; }
        tr:hover { background: #f0f7ff; }
        .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; margin: 15px 0; }
        .metric { background: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center; }
        .metric-value { font-size: 24px; font-weight: bold; color: #1a1a2e; }
        .metric-label { font-size: 11px; color: #666; text-transform: uppercase; margin-top: 5px; }
        .two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        .generated { color: #888; font-size: 12px; }
        .positive { color: #4CAF50; }
        .negative { color: #FF6B6B; }
    </style>
</head>
<body>
<div class="container">
    <h1>📊 Portfolio Analysis Report</h1>
    <p class="generated">Generated: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
""")

    if portfolios and selected_idx < len(portfolios):
        selected = portfolios[selected_idx]
        selected_strategies = [strategies[i] for i in selected.strategy_indices if i < len(strategies)]

        # ==================== Portfolio Summary ====================
        html_parts.append(f"""
    <div class="card">
        <h2>Portfolio #{selected_idx + 1} Summary</h2>
        <div class="metrics-grid">
            <div class="metric">
                <div class="metric-value">${selected.total_profit:,.2f}</div>
                <div class="metric-label">Total Profit</div>
            </div>
            <div class="metric">
                <div class="metric-value">${selected.equity_dd:,.2f}</div>
                <div class="metric-label">Equity DD</div>
            </div>
            <div class="metric">
                <div class="metric-value">${selected.balance_dd:,.2f}</div>
                <div class="metric-label">Balance DD</div>
            </div>
            <div class="metric">
                <div class="metric-value">{selected.return_dd_ratio:.2f}</div>
                <div class="metric-label">Return/DD</div>
            </div>
            <div class="metric">
                <div class="metric-value">{selected.sharpe:.2f}</div>
                <div class="metric-label">Avg Sharpe</div>
            </div>
            <div class="metric">
                <div class="metric-value">{selected.avg_correlation:.3f}</div>
                <div class="metric-label">Avg Correlation</div>
            </div>
        </div>
    </div>
""")

        # Build equity curve and calculate advanced stats
        equity_df = build_equity_curve(selected_strategies)
        adv_stats = calculate_advanced_stats(selected_strategies) if selected_strategies else {}

        # ==================== Advanced Statistics ====================
        if adv_stats:
            html_parts.append(f"""
    <div class="card">
        <h2>Advanced Statistics</h2>
        <div class="metrics-grid">
            <div class="metric">
                <div class="metric-value">{adv_stats.get('cagr', 0)*100:.2f}%</div>
                <div class="metric-label">CAGR</div>
            </div>
            <div class="metric">
                <div class="metric-value">{adv_stats.get('sqn', 0):.2f}</div>
                <div class="metric-label">SQN Score</div>
            </div>
            <div class="metric">
                <div class="metric-value">{adv_stats.get('stability', 0):.2f}</div>
                <div class="metric-label">Stability (R²)</div>
            </div>
            <div class="metric">
                <div class="metric-value">{adv_stats.get('symmetry', 0)*100:.1f}%</div>
                <div class="metric-label">Symmetry</div>
            </div>
            <div class="metric">
                <div class="metric-value">{adv_stats.get('z_score', 0):.2f}</div>
                <div class="metric-label">Z-Score</div>
            </div>
            <div class="metric">
                <div class="metric-value">{adv_stats.get('r_expectancy', 0):.2f}R</div>
                <div class="metric-label">R-Expectancy</div>
            </div>
            <div class="metric">
                <div class="metric-value">{adv_stats.get('profit_factor', 0):.2f}</div>
                <div class="metric-label">Profit Factor</div>
            </div>
            <div class="metric">
                <div class="metric-value">{adv_stats.get('win_rate', 0)*100:.1f}%</div>
                <div class="metric-label">Win Rate</div>
            </div>
            <div class="metric">
                <div class="metric-value">${adv_stats.get('avg_trade', 0):.2f}</div>
                <div class="metric-label">Avg Trade</div>
            </div>
            <div class="metric">
                <div class="metric-value">{adv_stats.get('total_trades', 0)}</div>
                <div class="metric-label">Total Trades</div>
            </div>
        </div>
    </div>
""")

            # ==================== Trade Statistics ====================
            html_parts.append(f"""
    <div class="card">
        <h2>Trade Statistics</h2>
        <div class="two-col">
            <div>
                <h3>Wins & Losses</h3>
                <table>
                    <tr><td>Number of Wins</td><td><strong>{adv_stats.get('num_wins', 0)}</strong></td></tr>
                    <tr><td>Number of Losses</td><td><strong>{adv_stats.get('num_losses', 0)}</strong></td></tr>
                    <tr><td>Gross Profit</td><td class="positive"><strong>${adv_stats.get('gross_profit', 0):,.2f}</strong></td></tr>
                    <tr><td>Gross Loss</td><td class="negative"><strong>${adv_stats.get('gross_loss', 0):,.2f}</strong></td></tr>
                    <tr><td>Average Win</td><td class="positive"><strong>${adv_stats.get('avg_win', 0):,.2f}</strong></td></tr>
                    <tr><td>Average Loss</td><td class="negative"><strong>${adv_stats.get('avg_loss', 0):,.2f}</strong></td></tr>
                    <tr><td>Largest Win</td><td class="positive"><strong>${adv_stats.get('largest_win', 0):,.2f}</strong></td></tr>
                    <tr><td>Largest Loss</td><td class="negative"><strong>${adv_stats.get('largest_loss', 0):,.2f}</strong></td></tr>
                </table>
            </div>
            <div>
                <h3>Trade Breakdown</h3>
                <table>
                    <tr><td>Max Consecutive Wins</td><td><strong>{adv_stats.get('max_consec_wins', 0)}</strong></td></tr>
                    <tr><td>Max Consecutive Losses</td><td><strong>{adv_stats.get('max_consec_losses', 0)}</strong></td></tr>
                    <tr><td>Long Trades</td><td><strong>{adv_stats.get('long_trades', 0)}</strong></td></tr>
                    <tr><td>Short Trades</td><td><strong>{adv_stats.get('short_trades', 0)}</strong></td></tr>
                    <tr><td>Long Profit</td><td><strong>${adv_stats.get('long_profit', 0):,.2f}</strong></td></tr>
                    <tr><td>Short Profit</td><td><strong>${adv_stats.get('short_profit', 0):,.2f}</strong></td></tr>
                    <tr><td>Stagnation Period</td><td><strong>{adv_stats.get('stagnation_days', 0)} days</strong></td></tr>
                </table>
            </div>
        </div>
    </div>
""")

        # ==================== Equity Curve ====================
        if not equity_df.empty:
            fig_equity = go.Figure()
            fig_equity.add_trace(go.Scatter(
                x=equity_df['datetime'],
                y=equity_df['balance'],
                mode='lines',
                name='Balance',
                line=dict(color='#4472C4', width=2)
            ))
            fig_equity.update_layout(
                title='Portfolio Equity Curve',
                xaxis_title='Date',
                yaxis_title='Balance ($)',
                template='plotly_white',
                height=400
            )

            html_parts.append(f"""
    <div class="card">
        <h2>Equity Curve</h2>
        <div id="equity-chart"></div>
        <script>
            var equityData = {fig_equity.to_json()};
            Plotly.newPlot('equity-chart', equityData.data, equityData.layout);
        </script>
    </div>
""")

            # ==================== Drawdown Chart ====================
            # Calculate drawdown properly: negative values showing decline from peak
            balance_series = equity_df['balance'].values
            running_peak = np.maximum.accumulate(balance_series)
            drawdown_values = balance_series - running_peak  # Should be 0 or negative

            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(
                x=equity_df['datetime'],
                y=drawdown_values,
                mode='lines',
                fill='tozeroy',
                name='Drawdown',
                line=dict(color='#FF6B6B'),
                fillcolor='rgba(255, 107, 107, 0.3)'
            ))
            fig_dd.update_layout(
                title='Portfolio Drawdown (Decline from Peak)',
                xaxis_title='Date',
                yaxis_title='Drawdown ($)',
                template='plotly_white',
                height=300,
                yaxis=dict(range=[min(drawdown_values) * 1.1 if min(drawdown_values) < 0 else -100, 10])
            )

            html_parts.append(f"""
    <div class="card">
        <h2>Drawdown Analysis</h2>
        <div id="dd-chart"></div>
        <script>
            var ddData = {fig_dd.to_json()};
            Plotly.newPlot('dd-chart', ddData.data, ddData.layout);
        </script>
    </div>
""")

            # ==================== P/L by Hour ====================
            if adv_stats and 'pl_by_hour' in adv_stats:
                pl_by_hour = adv_stats['pl_by_hour']
                hours = list(range(24))
                hour_profits = [pl_by_hour.get(h, 0) for h in hours]
                colors_hour = ['#4CAF50' if p > 0 else '#FF6B6B' for p in hour_profits]

                fig_hour = go.Figure(data=[go.Bar(
                    x=hours,
                    y=hour_profits,
                    marker_color=colors_hour,
                    text=[f"${p:,.0f}" for p in hour_profits],
                    textposition='outside'
                )])
                fig_hour.update_layout(
                    title='Profit/Loss by Hour of Day',
                    xaxis_title='Hour (0-23)',
                    yaxis_title='Profit/Loss ($)',
                    template='plotly_white',
                    height=350
                )

                html_parts.append(f"""
    <div class="card">
        <h2>P/L by Hour</h2>
        <div id="hour-chart"></div>
        <script>
            var hourData = {fig_hour.to_json()};
            Plotly.newPlot('hour-chart', hourData.data, hourData.layout);
        </script>
    </div>
""")

            # ==================== P/L by Day of Week ====================
            if adv_stats and 'pl_by_day' in adv_stats:
                pl_by_day = adv_stats['pl_by_day']
                day_order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
                day_profits = [pl_by_day.get(d, 0) for d in day_order]
                colors_day = ['#4CAF50' if p > 0 else '#FF6B6B' for p in day_profits]

                fig_day = go.Figure(data=[go.Bar(
                    x=day_order,
                    y=day_profits,
                    marker_color=colors_day,
                    text=[f"${p:,.0f}" for p in day_profits],
                    textposition='outside'
                )])
                fig_day.update_layout(
                    title='Profit/Loss by Day of Week',
                    xaxis_title='Day',
                    yaxis_title='Profit/Loss ($)',
                    template='plotly_white',
                    height=350
                )

                html_parts.append(f"""
    <div class="card">
        <h2>P/L by Day of Week</h2>
        <div id="day-chart"></div>
        <script>
            var dayData = {fig_day.to_json()};
            Plotly.newPlot('day-chart', dayData.data, dayData.layout);
        </script>
    </div>
""")

            # ==================== Monthly Returns ====================
            try:
                equity_df_monthly = equity_df.copy()
                equity_df_monthly['year_month'] = equity_df_monthly['datetime'].dt.to_period('M').astype(str)

                monthly = equity_df_monthly.groupby('year_month')['profit'].sum().reset_index()
                monthly = monthly.sort_values('year_month')

                if len(monthly) > 0:
                    # Monthly bar chart
                    colors = ['#4CAF50' if p > 0 else '#FF6B6B' for p in monthly['profit']]

                    fig_monthly = go.Figure(data=[go.Bar(
                        x=monthly['year_month'],
                        y=monthly['profit'],
                        marker_color=colors,
                        text=[f"${p:,.0f}" for p in monthly['profit']],
                        textposition='outside'
                    )])
                    fig_monthly.update_layout(
                        title='Monthly P/L',
                        xaxis_title='Month',
                        yaxis_title='Profit/Loss ($)',
                        template='plotly_white',
                        height=350
                    )

                    # Calculate stats
                    avg_monthly = monthly['profit'].mean()
                    best_month = monthly['profit'].max()
                    worst_month = monthly['profit'].min()
                    win_months = (monthly['profit'] > 0).sum()
                    total_months = len(monthly)

                    html_parts.append(f"""
    <div class="card">
        <h2>Monthly Returns</h2>
        <div class="metrics-grid">
            <div class="metric">
                <div class="metric-value">${avg_monthly:,.0f}</div>
                <div class="metric-label">Avg Monthly P/L</div>
            </div>
            <div class="metric">
                <div class="metric-value positive">${best_month:,.0f}</div>
                <div class="metric-label">Best Month</div>
            </div>
            <div class="metric">
                <div class="metric-value negative">${worst_month:,.0f}</div>
                <div class="metric-label">Worst Month</div>
            </div>
            <div class="metric">
                <div class="metric-value">{win_months}/{total_months} ({100*win_months/total_months:.0f}%)</div>
                <div class="metric-label">Profitable Months</div>
            </div>
        </div>
        <div id="monthly-chart"></div>
        <script>
            var monthlyData = {fig_monthly.to_json()};
            Plotly.newPlot('monthly-chart', monthlyData.data, monthlyData.layout);
        </script>
    </div>
""")
            except Exception as e:
                # Skip monthly section if there's an error
                pass

        # ==================== Strategies in Portfolio ====================
        html_parts.append("""
    <div class="card">
        <h2>Strategies in Portfolio</h2>
        <table>
            <tr>
                <th>Name</th>
                <th>Symbol</th>
                <th>Net Profit</th>
                <th>Max DD</th>
                <th>Sharpe</th>
                <th>Profit Factor</th>
                <th>Win Rate</th>
                <th>Trades</th>
            </tr>
""")

        for s in selected_strategies:
            html_parts.append(f"""
            <tr>
                <td>{s.name}</td>
                <td>{s.symbol}</td>
                <td>${s.metrics.total_net_profit:,.2f}</td>
                <td>${s.metrics.equity_dd_maximal:,.2f}</td>
                <td>{s.metrics.sharpe_ratio:.2f}</td>
                <td>{s.metrics.profit_factor:.2f}</td>
                <td>{s.metrics.win_rate:.1f}%</td>
                <td>{s.metrics.total_trades}</td>
            </tr>
""")

        html_parts.append("""
        </table>
    </div>
""")

    # ==================== Monte Carlo ====================
    if monte_carlo_result:
        html_parts.append(f"""
    <div class="card">
        <h2>Monte Carlo Simulation</h2>
        <div class="metrics-grid">
            <div class="metric">
                <div class="metric-value">{monte_carlo_result.get('num_simulations', 'N/A')}</div>
                <div class="metric-label">Simulations</div>
            </div>
            <div class="metric">
                <div class="metric-value">${monte_carlo_result.get('mean_final_balance', 0):,.2f}</div>
                <div class="metric-label">Mean Final Balance</div>
            </div>
            <div class="metric">
                <div class="metric-value">${monte_carlo_result.get('median_final_balance', 0):,.2f}</div>
                <div class="metric-label">Median Final Balance</div>
            </div>
            <div class="metric">
                <div class="metric-value">${monte_carlo_result.get('percentile_5', 0):,.2f}</div>
                <div class="metric-label">5th Percentile</div>
            </div>
            <div class="metric">
                <div class="metric-value">${monte_carlo_result.get('percentile_95', 0):,.2f}</div>
                <div class="metric-label">95th Percentile</div>
            </div>
            <div class="metric">
                <div class="metric-value">{monte_carlo_result.get('risk_of_ruin_50', 0):.1%}</div>
                <div class="metric-label">Risk of 50% DD</div>
            </div>
        </div>
    </div>
""")

    # Footer
    html_parts.append("""
</div>
</body>
</html>
""")

    return ''.join(html_parts)



def init_session_state():
    """Initialize session state variables."""
    if 'strategies' not in st.session_state:
        st.session_state.strategies = []  # List of Strategy objects
    if 'locked_indices' not in st.session_state:
        st.session_state.locked_indices = set()  # Indices of locked strategies
    if 'portfolios' not in st.session_state:
        st.session_state.portfolios = []  # Generated portfolios
    if 'optimization_running' not in st.session_state:
        st.session_state.optimization_running = False
    if 'selected_portfolio' not in st.session_state:
        st.session_state.selected_portfolio = None
    if 'correlation_matrix' not in st.session_state:
        st.session_state.correlation_matrix = None
    if 'strategy_names' not in st.session_state:
        st.session_state.strategy_names = []
    if 'monte_carlo_result' not in st.session_state:
        st.session_state.monte_carlo_result = None
    if 'dark_mode' not in st.session_state:
        st.session_state.dark_mode = False  # Default to light theme
    if 'is_generating' not in st.session_state:
        st.session_state.is_generating = False  # Continuous generation flag
    if 'is_paused' not in st.session_state:
        st.session_state.is_paused = False  # Paused state for reviewing results
    if 'generation_count' not in st.session_state:
        st.session_state.generation_count = 0  # Number of iterations run
    if 'total_explored' not in st.session_state:
        st.session_state.total_explored = 0  # Total combinations explored


def parse_uploaded_files(uploaded_files) -> List[Strategy]:
    """Parse uploaded MT5 HTML files."""
    strategies = []
    parser = MT5Parser()

    for uploaded_file in uploaded_files:
        try:
            # Save to temp file and parse
            content = uploaded_file.read()

            # Try different encodings
            decoded_content = None
            for encoding in ['utf-16', 'utf-8', 'latin-1']:
                try:
                    decoded_content = content.decode(encoding)
                    break
                except:
                    continue

            if decoded_content is None:
                st.warning(f"Could not decode {uploaded_file.name}")
                continue

            # Parse the content
            strategy = parser._parse_content(decoded_content)
            strategy.file_path = uploaded_file.name
            strategies.append(strategy)

        except Exception as e:
            st.warning(f"Error parsing {uploaded_file.name}: {str(e)}")

    # Same-named reports (one EA, several parameter sets) must not collide.
    make_names_unique(strategies)
    return strategies


def strategy_to_dict(strategy: Strategy) -> Dict:
    """Convert Strategy to dictionary for DataFrame."""
    m = strategy.metrics
    return {
        'Name': strategy.name,
        'Symbol': strategy.symbol,
        'Timeframe': strategy.timeframe,
        'Net Profit': m.total_net_profit,
        'Equity DD $': m.equity_dd_maximal,
        'Equity DD %': m.equity_dd_maximal_pct,
        'Balance DD $': m.balance_dd_maximal,
        'Balance DD %': m.balance_dd_maximal_pct,
        'Sharpe': m.sharpe_ratio,
        'Profit Factor': m.profit_factor,
        'Recovery Factor': m.recovery_factor,
        'Return/DD': m.total_net_profit / m.equity_dd_maximal if m.equity_dd_maximal > 0 else 0,
        'Win Rate %': m.win_rate,
        'Total Trades': m.total_trades,
        'Initial Deposit': m.initial_deposit,
    }


def get_strategies_dataframe() -> pd.DataFrame:
    """Get strategies as a pandas DataFrame."""
    if not st.session_state.strategies:
        return pd.DataFrame()

    data = [strategy_to_dict(s) for s in st.session_state.strategies]
    df = pd.DataFrame(data)

    # Add locked status
    df['Locked'] = [i in st.session_state.locked_indices for i in range(len(df))]

    return df


def filter_strategies(df: pd.DataFrame, filters: Dict) -> pd.DataFrame:
    """Apply filters to strategies dataframe (individual strategy filters only)."""
    filtered = df.copy()

    # Note: DD filtering is now done at portfolio level, not individual strategy level

    # Other filters for individual strategies
    if filters.get('min_sharpe', 0) > 0:
        filtered = filtered[filtered['Sharpe'] >= filters['min_sharpe']]

    if filters.get('min_profit_factor', 0) > 0:
        filtered = filtered[filtered['Profit Factor'] >= filters['min_profit_factor']]

    if filters.get('min_trades', 0) > 0:
        filtered = filtered[filtered['Total Trades'] >= filters['min_trades']]

    return filtered


@dataclass
class Portfolio:
    """Represents a portfolio of strategies."""
    strategy_indices: List[int]
    strategy_names: List[str]
    total_profit: float
    balance_dd: float  # Actual portfolio balance DD from merged trades
    equity_dd: float   # Estimated portfolio equity DD
    return_dd_ratio: float
    avg_correlation: float
    sharpe: float
    symbols: Dict[str, int]  # Symbol -> count
    stability: float = 0.0  # R² of equity curve (1.0 = perfect straight line)
    symmetry: float = 0.0   # Profit distribution consistency


def calculate_portfolio_balance_dd(strategies: List[Strategy]) -> Tuple[float, List[Tuple[str, float]]]:
    """
    Calculate actual portfolio Balance DD from merged trade data.

    Returns:
        Tuple of (max_drawdown_dollars, equity_curve as list of (date, cumulative_profit))
    """
    # Merge all closed trades from all strategies
    all_trades = []
    for strategy in strategies:
        for trade in strategy.trades:
            if trade.direction == "out" and trade.profit != 0:
                all_trades.append((trade.time, trade.profit))

    if not all_trades:
        return 0.0, []

    # Sort by time
    all_trades.sort(key=lambda x: x[0])

    # Build cumulative balance curve
    cumulative = 0.0
    peak = 0.0
    max_dd = 0.0
    equity_curve = []

    for trade_time, profit in all_trades:
        cumulative += profit
        equity_curve.append((trade_time.strftime("%Y-%m-%d"), cumulative))

        if cumulative > peak:
            peak = cumulative

        drawdown = peak - cumulative
        if drawdown > max_dd:
            max_dd = drawdown

    return max_dd, equity_curve


def estimate_portfolio_equity_dd(
    strategies: List[Strategy],
    avg_correlation: float
) -> float:
    """
    Estimate portfolio Equity DD using correlation-adjusted calculation.

    Since we don't have tick-by-tick data, we estimate using:
    - Sum of individual equity DDs
    - Adjusted by correlation factor (lower correlation = more diversification benefit)

    Formula: Portfolio_DD = sqrt(sum(DD_i^2) + 2*sum(DD_i*DD_j*corr_ij))
    Simplified: Portfolio_DD ≈ sum(DD_i) * sqrt((1 + (n-1)*avg_corr) / n)
    """
    if not strategies:
        return 0.0

    individual_dds = [s.metrics.equity_dd_maximal for s in strategies]
    n = len(strategies)

    if n == 1:
        return individual_dds[0]

    # Sum of individual DDs
    total_dd = sum(individual_dds)

    # Diversification factor based on correlation
    # When corr = 1.0, factor = 1.0 (no diversification benefit)
    # When corr = 0.0, factor = 1/sqrt(n) (maximum diversification)
    # When corr = -1.0, factor approaches 0 (perfect hedge)
    avg_corr = max(-1.0, min(1.0, avg_correlation))  # Clamp to valid range

    # Calculate diversification factor
    diversification_factor = np.sqrt((1 + (n - 1) * avg_corr) / n)

    # Estimated portfolio equity DD
    estimated_dd = total_dd * diversification_factor

    return estimated_dd


def calculate_portfolio_metrics(
    indices: List[int],
    strategies: List[Strategy],
    correlation_engine: Optional[CorrelationEngine] = None
) -> Portfolio:
    """Calculate combined metrics for a portfolio."""
    selected = [strategies[i] for i in indices]

    # Basic metrics
    total_profit = sum(s.metrics.total_net_profit for s in selected)

    # Calculate average correlation first (needed for equity DD estimate)
    avg_corr = 0.0
    if correlation_engine and len(selected) > 1:
        correlations = []
        for i, s1 in enumerate(selected):
            for s2 in selected[i+1:]:
                corr = correlation_engine.calculate_correlation(s1, s2).correlation
                correlations.append(corr)
        avg_corr = np.mean(correlations) if correlations else 0.0

    # Calculate actual Balance DD from merged trades
    balance_dd, _ = calculate_portfolio_balance_dd(selected)

    # Estimate Equity DD using correlation-adjusted calculation
    equity_dd = estimate_portfolio_equity_dd(selected, avg_corr)

    # Use equity DD for return ratio (more conservative)
    return_dd = total_profit / equity_dd if equity_dd > 0 else 0

    # Symbol distribution
    symbols = {}
    for s in selected:
        symbols[s.symbol] = symbols.get(s.symbol, 0) + 1

    # Combined Sharpe (simplified average)
    avg_sharpe = np.mean([s.metrics.sharpe_ratio for s in selected])

    # Calculate stability and symmetry
    stability = calculate_portfolio_stability(selected)
    symmetry = calculate_portfolio_symmetry(selected)

    return Portfolio(
        strategy_indices=indices,
        strategy_names=[s.name for s in selected],
        total_profit=total_profit,
        balance_dd=balance_dd,
        equity_dd=equity_dd,
        return_dd_ratio=return_dd,
        avg_correlation=avg_corr,
        sharpe=avg_sharpe,
        symbols=symbols,
        stability=stability,
        symmetry=symmetry
    )


def generate_portfolios(
    strategies: List[Strategy],
    locked_indices: set,
    filters: Dict,
    settings: Dict,
    progress_callback=None,
    max_iterations: int = 50000  # Limit iterations for tractability
) -> List[Portfolio]:
    """Generate and rank portfolios based on settings using random sampling."""
    import random

    # Build correlation engine
    correlation_engine = CorrelationEngine(strategies)
    correlation_engine.build_correlation_matrix()

    # Get filtered strategy indices (excluding locked - they're always included)
    df = get_strategies_dataframe()
    filtered_df = filter_strategies(df, filters)
    filtered_indices = set(filtered_df.index.tolist())

    # Available candidates (filtered and not locked)
    candidate_indices = list(filtered_indices - locked_indices)

    # Determine how many to add
    locked_count = len(locked_indices)
    min_to_add = max(0, settings['min_strategies'] - locked_count)
    max_to_add = settings['max_strategies'] - locked_count

    if max_to_add < 0:
        st.warning(f"Already have {locked_count} locked strategies, which exceeds max portfolio size of {settings['max_strategies']}")
        return []

    if not candidate_indices:
        st.warning("No candidate strategies pass the filters")
        return []

    portfolios = []
    seen_combos = set()  # Track unique combinations

    # Calculate total combinations to see if we can enumerate all
    from math import comb
    total_combinations = 0
    for n in range(min_to_add, max_to_add + 1):
        if n <= len(candidate_indices):
            total_combinations += comb(len(candidate_indices), n)

    # If total combinations is small enough, enumerate all
    use_exhaustive = total_combinations <= max_iterations

    if use_exhaustive:
        # Exhaustive search
        processed = 0
        for n_add in range(min_to_add, max_to_add + 1):
            if n_add > len(candidate_indices):
                continue

            for combo in combinations(candidate_indices, n_add):
                portfolio_indices = list(locked_indices) + list(combo)

                # Check max per symbol constraint
                symbols_count = {}
                valid = True
                for idx in portfolio_indices:
                    symbol = strategies[idx].symbol
                    symbols_count[symbol] = symbols_count.get(symbol, 0) + 1
                    if symbols_count[symbol] > settings['max_per_symbol']:
                        valid = False
                        break

                if not valid:
                    processed += 1
                    if progress_callback and processed % 100 == 0:
                        progress_callback(processed / total_combinations)
                    continue

                portfolio = calculate_portfolio_metrics(
                    portfolio_indices, strategies, correlation_engine
                )

                # Check correlation constraint
                if settings['max_correlation'] < 1.0 and portfolio.avg_correlation > settings['max_correlation']:
                    processed += 1
                    if progress_callback and processed % 100 == 0:
                        progress_callback(processed / total_combinations)
                    continue

                # Check portfolio DD constraints
                max_equity_dd = settings.get('max_portfolio_equity_dd', 0)
                max_balance_dd = settings.get('max_portfolio_balance_dd', 0)

                if max_equity_dd > 0 and portfolio.equity_dd > max_equity_dd:
                    processed += 1
                    if progress_callback and processed % 100 == 0:
                        progress_callback(processed / total_combinations)
                    continue

                if max_balance_dd > 0 and portfolio.balance_dd > max_balance_dd:
                    processed += 1
                    if progress_callback and processed % 100 == 0:
                        progress_callback(processed / total_combinations)
                    continue

                portfolios.append(portfolio)
                processed += 1

                if progress_callback and processed % 100 == 0:
                    progress_callback(processed / total_combinations)

    else:
        # Random sampling approach for large search spaces
        st.info(f"Search space too large ({total_combinations:,} combinations). Using smart random sampling ({max_iterations:,} samples)...")

        for i in range(max_iterations):
            # Random portfolio size
            n_add = random.randint(min_to_add, min(max_to_add, len(candidate_indices)))

            if n_add == 0:
                combo = tuple()
            else:
                combo = tuple(sorted(random.sample(candidate_indices, n_add)))

            # Skip if we've seen this combination
            if combo in seen_combos:
                continue
            seen_combos.add(combo)

            portfolio_indices = list(locked_indices) + list(combo)

            # Check max per symbol constraint
            symbols_count = {}
            valid = True
            for idx in portfolio_indices:
                symbol = strategies[idx].symbol
                symbols_count[symbol] = symbols_count.get(symbol, 0) + 1
                if symbols_count[symbol] > settings['max_per_symbol']:
                    valid = False
                    break

            if not valid:
                continue

            portfolio = calculate_portfolio_metrics(
                portfolio_indices, strategies, correlation_engine
            )

            # Check correlation constraint
            if settings['max_correlation'] < 1.0 and portfolio.avg_correlation > settings['max_correlation']:
                continue

            # Check portfolio DD constraints
            max_equity_dd = settings.get('max_portfolio_equity_dd', 0)
            max_balance_dd = settings.get('max_portfolio_balance_dd', 0)

            if max_equity_dd > 0 and portfolio.equity_dd > max_equity_dd:
                continue

            if max_balance_dd > 0 and portfolio.balance_dd > max_balance_dd:
                continue

            portfolios.append(portfolio)

            # Keep only top N * 2 during search to save memory
            if len(portfolios) > settings['keep_top'] * 2:
                if settings['rank_by'] == 'Return/DD':
                    portfolios.sort(key=lambda p: p.return_dd_ratio, reverse=True)
                elif settings['rank_by'] == 'Sharpe':
                    portfolios.sort(key=lambda p: p.sharpe, reverse=True)
                elif settings['rank_by'] == 'Net Profit':
                    portfolios.sort(key=lambda p: p.total_profit, reverse=True)
                elif settings['rank_by'] == 'Lowest DD':
                    portfolios.sort(key=lambda p: p.equity_dd, reverse=False)
                elif settings['rank_by'] == 'Lowest Correlation':
                    portfolios.sort(key=lambda p: p.avg_correlation, reverse=False)
                portfolios = portfolios[:settings['keep_top']]

            if progress_callback and i % 500 == 0:
                progress_callback(i / max_iterations)

    # Final sort
    if settings['rank_by'] == 'Return/DD':
        portfolios.sort(key=lambda p: p.return_dd_ratio, reverse=True)
    elif settings['rank_by'] == 'Sharpe':
        portfolios.sort(key=lambda p: p.sharpe, reverse=True)
    elif settings['rank_by'] == 'Net Profit':
        portfolios.sort(key=lambda p: p.total_profit, reverse=True)
    elif settings['rank_by'] == 'Lowest DD':
        portfolios.sort(key=lambda p: p.equity_dd, reverse=False)
    elif settings['rank_by'] == 'Lowest Correlation':
        portfolios.sort(key=lambda p: p.avg_correlation, reverse=False)

    return portfolios[:settings['keep_top']]


def generate_portfolios_genetic(
    strategies: List[Strategy],
    locked_indices: set,
    filters: Dict,
    settings: Dict,
    progress_callback=None,
    population_size: int = 100,
    generations: int = 50,
    mutation_rate: float = 0.1
) -> List[Portfolio]:
    """Generate portfolios using genetic algorithm for better convergence in large search spaces."""
    import random

    # Build correlation engine
    correlation_engine = CorrelationEngine(strategies)
    correlation_engine.build_correlation_matrix()

    # Get filtered strategy indices
    df = get_strategies_dataframe()
    filtered_df = filter_strategies(df, filters)
    filtered_indices = set(filtered_df.index.tolist())
    candidate_indices = list(filtered_indices - locked_indices)

    # Determine portfolio size range
    locked_count = len(locked_indices)
    min_to_add = max(0, settings['min_strategies'] - locked_count)
    max_to_add = settings['max_strategies'] - locked_count

    if max_to_add < 0 or not candidate_indices:
        return []

    def create_random_individual():
        """Create a random portfolio (individual)."""
        n_add = random.randint(min_to_add, min(max_to_add, len(candidate_indices)))
        if n_add == 0:
            return list(locked_indices)
        genes = random.sample(candidate_indices, n_add)
        return list(locked_indices) + genes

    def is_valid(individual):
        """Check if portfolio meets constraints."""
        # Check symbol constraint
        symbols_count = {}
        for idx in individual:
            symbol = strategies[idx].symbol
            symbols_count[symbol] = symbols_count.get(symbol, 0) + 1
            if symbols_count[symbol] > settings['max_per_symbol']:
                return False
        return True

    def get_fitness(portfolio):
        """Get fitness score based on rank_by setting."""
        if settings['rank_by'] == 'Return/DD':
            return portfolio.return_dd_ratio
        elif settings['rank_by'] == 'Sharpe':
            return portfolio.sharpe
        elif settings['rank_by'] == 'Net Profit':
            return portfolio.total_profit
        elif settings['rank_by'] == 'Lowest DD':
            return -portfolio.equity_dd  # Negative because we want to minimize
        elif settings['rank_by'] == 'Lowest Correlation':
            return -portfolio.avg_correlation
        return portfolio.return_dd_ratio

    def evaluate_individual(individual):
        """Evaluate an individual and return Portfolio if valid, None otherwise."""
        if not is_valid(individual):
            return None

        portfolio = calculate_portfolio_metrics(individual, strategies, correlation_engine)

        # Check constraints
        if settings['max_correlation'] < 1.0 and portfolio.avg_correlation > settings['max_correlation']:
            return None

        max_equity_dd = settings.get('max_portfolio_equity_dd', 0)
        max_balance_dd = settings.get('max_portfolio_balance_dd', 0)

        if max_equity_dd > 0 and portfolio.equity_dd > max_equity_dd:
            return None
        if max_balance_dd > 0 and portfolio.balance_dd > max_balance_dd:
            return None

        return portfolio

    def crossover(parent1, parent2):
        """Create child by combining strategies from two parents."""
        # Get strategies not in locked set
        genes1 = [i for i in parent1 if i not in locked_indices]
        genes2 = [i for i in parent2 if i not in locked_indices]

        # Combine and deduplicate
        all_genes = list(set(genes1 + genes2))

        # Random selection from combined pool
        n_select = random.randint(min_to_add, min(max_to_add, len(all_genes)))
        if n_select > 0 and all_genes:
            child_genes = random.sample(all_genes, min(n_select, len(all_genes)))
        else:
            child_genes = []

        return list(locked_indices) + child_genes

    def mutate(individual):
        """Mutate by swapping one strategy."""
        genes = [i for i in individual if i not in locked_indices]
        available = [i for i in candidate_indices if i not in individual]

        if genes and available and random.random() < mutation_rate:
            # Remove one random gene
            to_remove = random.choice(genes)
            genes.remove(to_remove)
            # Add one random new gene
            to_add = random.choice(available)
            genes.append(to_add)

        return list(locked_indices) + genes

    # Initialize population
    population = []
    seen = set()

    for _ in range(population_size * 3):  # Try more to get enough valid ones
        individual = create_random_individual()
        key = tuple(sorted(individual))
        if key not in seen:
            seen.add(key)
            portfolio = evaluate_individual(individual)
            if portfolio:
                population.append((individual, portfolio))
        if len(population) >= population_size:
            break

    if not population:
        return []

    best_portfolios = []
    total_steps = generations

    # Evolution loop
    for gen in range(generations):
        # Sort by fitness
        population.sort(key=lambda x: get_fitness(x[1]), reverse=True)

        # Keep top performers
        elite_size = max(2, population_size // 10)
        elite = population[:elite_size]

        # Track best
        for ind, port in elite:
            key = tuple(sorted(ind))
            if key not in seen or port not in best_portfolios:
                best_portfolios.append(port)

        # Create next generation
        new_population = list(elite)  # Keep elite

        while len(new_population) < population_size:
            # Tournament selection
            tournament_size = 3
            tournament = random.sample(population[:population_size//2], min(tournament_size, len(population[:population_size//2])))
            parent1 = max(tournament, key=lambda x: get_fitness(x[1]))[0]

            tournament = random.sample(population[:population_size//2], min(tournament_size, len(population[:population_size//2])))
            parent2 = max(tournament, key=lambda x: get_fitness(x[1]))[0]

            # Crossover
            child = crossover(parent1, parent2)

            # Mutate
            child = mutate(child)

            # Evaluate
            key = tuple(sorted(child))
            if key not in seen:
                seen.add(key)
                portfolio = evaluate_individual(child)
                if portfolio:
                    new_population.append((child, portfolio))

        population = new_population

        if progress_callback:
            progress_callback((gen + 1) / total_steps)

    # Collect all valid portfolios found
    all_portfolios = [p for _, p in population] + best_portfolios

    # Remove duplicates
    seen_keys = set()
    unique_portfolios = []
    for p in all_portfolios:
        key = tuple(sorted(p.strategy_indices))
        if key not in seen_keys:
            seen_keys.add(key)
            unique_portfolios.append(p)

    # Final sort
    if settings['rank_by'] == 'Return/DD':
        unique_portfolios.sort(key=lambda p: p.return_dd_ratio, reverse=True)
    elif settings['rank_by'] == 'Sharpe':
        unique_portfolios.sort(key=lambda p: p.sharpe, reverse=True)
    elif settings['rank_by'] == 'Net Profit':
        unique_portfolios.sort(key=lambda p: p.total_profit, reverse=True)
    elif settings['rank_by'] == 'Lowest DD':
        unique_portfolios.sort(key=lambda p: p.equity_dd, reverse=False)
    elif settings['rank_by'] == 'Lowest Correlation':
        unique_portfolios.sort(key=lambda p: p.avg_correlation, reverse=False)

    return unique_portfolios[:settings['keep_top']]


def plot_equity_curves(strategies: List[Strategy], title: str = "Equity Curves"):
    """Plot equity curves for strategies."""
    fig = go.Figure()

    colors = px.colors.qualitative.Set2

    for i, strategy in enumerate(strategies):
        daily_returns = strategy.get_daily_returns()
        if not daily_returns:
            continue

        # Build cumulative equity
        dates = sorted(daily_returns.keys())
        cumulative = []
        running_total = strategy.metrics.initial_deposit or 10000

        for date in dates:
            running_total += daily_returns[date]
            cumulative.append(running_total)

        fig.add_trace(go.Scatter(
            x=dates,
            y=cumulative,
            name=strategy.name[:30],
            line=dict(color=colors[i % len(colors)], width=2),
            hovertemplate=f"{strategy.name}<br>Date: %{{x}}<br>Equity: $%{{y:,.2f}}<extra></extra>"
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Equity ($)",
        template="plotly_dark",
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    return fig


def plot_correlation_heatmap(strategies: List[Strategy]):
    """Plot correlation heatmap."""
    if len(strategies) < 2:
        return None

    engine = CorrelationEngine(strategies)
    matrix, names = engine.build_correlation_matrix()

    # Truncate names for display
    display_names = [n[:20] + '...' if len(n) > 20 else n for n in names]

    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=display_names,
        y=display_names,
        colorscale=[
            [0, '#3b82f6'],      # -1: Blue (negative correlation - good)
            [0.35, '#22c55e'],   # -0.3: Green
            [0.5, '#84cc16'],    # 0: Lime (no correlation - good)
            [0.65, '#f59e0b'],   # 0.3: Amber
            [1, '#ef4444']       # 1: Red (high correlation - bad)
        ],
        zmid=0,
        text=np.round(matrix, 2),
        texttemplate="%{text}",
        textfont={"size": 10},
        hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>Correlation: %{z:.3f}<extra></extra>"
    ))

    fig.update_layout(
        title="Correlation Matrix",
        template="plotly_dark",
        height=500,
        xaxis=dict(tickangle=45)
    )

    return fig


def plot_portfolio_composition(portfolio: Portfolio, strategies: List[Strategy]):
    """Plot portfolio composition breakdown."""
    selected = [strategies[i] for i in portfolio.strategy_indices]

    # Profit contribution
    profits = [s.metrics.total_net_profit for s in selected]
    names = [s.name[:20] for s in selected]

    fig = make_subplots(rows=1, cols=2, specs=[[{'type':'pie'}, {'type':'bar'}]],
                        subplot_titles=('Profit Contribution', 'Individual Metrics'))

    # Pie chart - profit contribution
    fig.add_trace(go.Pie(
        labels=names,
        values=[max(0, p) for p in profits],
        hole=0.4,
        textinfo='percent+label'
    ), row=1, col=1)

    # Bar chart - metrics comparison
    fig.add_trace(go.Bar(
        name='Sharpe',
        x=names,
        y=[s.metrics.sharpe_ratio for s in selected],
        marker_color='#3b82f6'
    ), row=1, col=2)

    fig.update_layout(
        template="plotly_dark",
        height=400,
        showlegend=False
    )

    return fig


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    init_session_state()

    # Apply theme based on session state
    apply_theme(st.session_state.dark_mode)

    st.title("📊 Portfolio Optimizer")
    st.markdown("Build and optimize trading strategy portfolios from MT5 backtest reports")

    # Sidebar - File Upload & Settings
    with st.sidebar:
        # Theme toggle at top of sidebar
        st.header("🎨 Theme")
        dark_mode = st.toggle("Dark Mode", value=st.session_state.dark_mode)
        if dark_mode != st.session_state.dark_mode:
            st.session_state.dark_mode = dark_mode
            st.rerun()

        st.divider()

        st.header("📁 Data Upload")

        uploaded_files = st.file_uploader(
            "Upload MT5 HTML Reports",
            type=['html', 'htm'],
            accept_multiple_files=True,
            help="Upload Strategy Tester HTML reports from MetaTrader 5"
        )

        if uploaded_files:
            if st.button("🔄 Parse Files", type="primary"):
                with st.spinner("Parsing files..."):
                    new_strategies = parse_uploaded_files(uploaded_files)
                    if new_strategies:
                        st.session_state.strategies.extend(new_strategies)
                        make_names_unique(st.session_state.strategies)  # across uploads too
                        st.success(f"Added {len(new_strategies)} strategies")
                        st.rerun()

        st.divider()

        # Strategy count
        st.metric("Total Strategies", len(st.session_state.strategies))
        st.metric("Locked Strategies", len(st.session_state.locked_indices))

        if st.session_state.strategies:
            if st.button("🗑️ Clear All", type="secondary"):
                st.session_state.strategies = []
                st.session_state.locked_indices = set()
                st.session_state.portfolios = []
                st.session_state.selected_portfolio = None
                st.rerun()

        st.divider()

        # Save/Load Databank
        st.header("💾 Databank")

        # Save section
        with st.expander("Save Databank", expanded=False):
            save_name = st.text_input(
                "Databank name",
                value=f"databank_{datetime.now().strftime('%Y%m%d')}",
                help="Enter a name for this databank"
            )
            if st.button("💾 Save", disabled=len(st.session_state.strategies) == 0):
                if st.session_state.strategies:
                    try:
                        filepath = save_databank(
                            save_name,
                            st.session_state.strategies,
                            st.session_state.locked_indices
                        )
                        st.success(f"Saved {len(st.session_state.strategies)} strategies!")
                    except Exception as e:
                        st.error(f"Error saving: {str(e)}")
                else:
                    st.warning("No strategies to save")

        # Load section
        saved_databanks = get_saved_databanks()
        with st.expander("Load Databank", expanded=False):
            if saved_databanks:
                selected_databank = st.selectbox(
                    "Select databank",
                    saved_databanks,
                    help="Choose a previously saved databank"
                )
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("📂 Load"):
                        try:
                            strategies, locked = load_databank(selected_databank)
                            st.session_state.strategies = strategies
                            st.session_state.locked_indices = locked
                            st.session_state.portfolios = []
                            st.success(f"Loaded {len(strategies)} strategies!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error loading: {str(e)}")
                with col2:
                    if st.button("🗑️ Delete"):
                        if delete_databank(selected_databank):
                            st.success(f"Deleted '{selected_databank}'")
                            st.rerun()
            else:
                st.info("No saved databanks yet")

    # Main content tabs
    if not st.session_state.strategies:
        st.info("👆 Upload MT5 HTML files using the sidebar to get started")

        # Show example
        st.markdown("""
        ### How to use:
        1. **Export** backtest reports from MT5 Strategy Tester as HTML
        2. **Upload** the HTML files using the sidebar
        3. **View** all strategies in the Databank tab
        4. **Filter** and configure portfolio settings
        5. **Build** optimal portfolios
        6. **Analyze** results with charts and metrics
        """)
        return

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Databank",
        "⚙️ Portfolio Builder",
        "🏆 Results",
        "📈 Analysis",
        "🎲 Monte Carlo"
    ])

    # ========================================================================
    # TAB 1: DATABANK
    # ========================================================================
    with tab1:
        st.header("Strategy Databank")

        df = get_strategies_dataframe()

        # Quick filters
        col1, col2, col3 = st.columns(3)
        with col1:
            sort_by = st.selectbox("Sort by", ['Return/DD', 'Net Profit', 'Sharpe', 'Profit Factor', 'Equity DD $'])
        with col2:
            sort_order = st.selectbox("Order", ['Descending', 'Ascending'])
        with col3:
            show_locked_only = st.checkbox("Show locked only")

        # Apply sorting
        ascending = sort_order == 'Ascending'
        if sort_by in df.columns:
            df_display = df.sort_values(sort_by, ascending=ascending)
        else:
            df_display = df

        if show_locked_only:
            df_display = df_display[df_display['Locked'] == True]

        # Display dataframe with selection
        st.markdown("**Click row to select/lock strategies:**")

        # Format the dataframe
        formatted_df = df_display.copy()
        formatted_df['Net Profit'] = formatted_df['Net Profit'].apply(lambda x: f"${x:,.2f}")
        formatted_df['Equity DD $'] = formatted_df['Equity DD $'].apply(lambda x: f"${x:,.2f}")
        formatted_df['Balance DD $'] = formatted_df['Balance DD $'].apply(lambda x: f"${x:,.2f}")
        formatted_df['Equity DD %'] = formatted_df['Equity DD %'].apply(lambda x: f"{x:.2f}%")
        formatted_df['Balance DD %'] = formatted_df['Balance DD %'].apply(lambda x: f"{x:.2f}%")
        formatted_df['Sharpe'] = formatted_df['Sharpe'].apply(lambda x: f"{x:.2f}")
        formatted_df['Profit Factor'] = formatted_df['Profit Factor'].apply(lambda x: f"{x:.2f}")
        formatted_df['Return/DD'] = formatted_df['Return/DD'].apply(lambda x: f"{x:.2f}")
        formatted_df['Win Rate %'] = formatted_df['Win Rate %'].apply(lambda x: f"{x:.1f}%")
        formatted_df['Locked'] = formatted_df['Locked'].apply(lambda x: '🔒' if x else '')

        st.dataframe(
            formatted_df,
            use_container_width=True,
            height=400
        )

        # Lock/Unlock controls
        st.subheader("Lock Strategies")

        strategy_options = [f"{i}: {s.name} ({s.symbol})" for i, s in enumerate(st.session_state.strategies)]

        col1, col2 = st.columns(2)

        with col1:
            to_lock = st.multiselect(
                "Select strategies to LOCK (keep in portfolio)",
                strategy_options,
                default=[strategy_options[i] for i in st.session_state.locked_indices if i < len(strategy_options)]
            )

            if st.button("🔒 Update Locked"):
                st.session_state.locked_indices = set()
                for item in to_lock:
                    idx = int(item.split(':')[0])
                    st.session_state.locked_indices.add(idx)
                st.success(f"Locked {len(st.session_state.locked_indices)} strategies")
                st.rerun()

        with col2:
            if st.button("🔓 Unlock All"):
                st.session_state.locked_indices = set()
                st.rerun()

    # ========================================================================
    # TAB 2: PORTFOLIO BUILDER
    # ========================================================================
    with tab2:
        st.header("Portfolio Builder Settings")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Strategy Filters")
            st.caption("Filter individual strategies before portfolio building")

            min_sharpe = st.number_input("Min Sharpe Ratio", min_value=0.0, value=0.0, step=0.1)
            min_pf = st.number_input("Min Profit Factor", min_value=0.0, value=1.0, step=0.1)
            min_trades = st.number_input("Min Total Trades", min_value=0, value=30, step=10)

            st.divider()

            st.subheader("Portfolio DD Limits")
            st.caption("Like QuantAnalyzer: filter portfolios by combined DD")

            max_portfolio_equity_dd = st.number_input(
                "Max Portfolio Equity DD ($)",
                min_value=0.0,
                value=500.0,
                step=50.0,
                help="Maximum estimated equity (open) drawdown for the portfolio"
            )
            max_portfolio_balance_dd = st.number_input(
                "Max Portfolio Balance DD ($)",
                min_value=0.0,
                value=600.0,
                step=50.0,
                help="Maximum balance (closed) drawdown for the portfolio"
            )

        with col2:
            st.subheader("Portfolio Settings")

            min_strategies = st.number_input("Min Strategies", min_value=1, value=5)
            max_strategies = st.number_input("Max Strategies", min_value=1, value=15)
            max_per_symbol = st.number_input("Max per Symbol", min_value=1, value=3,
                                             help="Limit strategies per symbol for diversification")
            max_correlation = st.slider("Max Avg Correlation", 0.0, 1.0, 0.7, 0.05)

            rank_by = st.selectbox("Rank Portfolios By",
                                   ['Return/DD', 'Sharpe', 'Net Profit', 'Lowest DD', 'Lowest Correlation'])
            keep_top = st.number_input("Keep Top N Portfolios", min_value=1, value=20)

        st.divider()

        # Mode selection
        mode = st.radio(
            "Optimization Mode",
            ["🆕 Create New Portfolio", "🔄 Add to Locked Strategies"],
            horizontal=True
        )

        if mode == "🔄 Add to Locked Strategies":
            if not st.session_state.locked_indices:
                st.warning("⚠️ No strategies locked. Go to Databank tab to lock strategies first.")
            else:
                st.success(f"Will keep {len(st.session_state.locked_indices)} locked strategies and find best additions")

        st.divider()

        # Search settings
        st.subheader("Search Settings")

        # Search algorithm selection
        search_algorithm = st.radio(
            "Search Algorithm",
            ["Random Sampling", "Genetic Search"],
            horizontal=True,
            help="Random: Fast, good for exploration. Genetic: Smarter, better for large search spaces.",
            disabled=(st.session_state.is_generating and not st.session_state.is_paused)
        )

        if search_algorithm == "Random Sampling":
            sampling_iterations = st.select_slider(
                "Combinations per Iteration",
                options=[10000, 25000, 50000, 100000, 250000, 500000, 1000000],
                value=100000,
                help="How many combinations to test per iteration.",
                disabled=(st.session_state.is_generating and not st.session_state.is_paused)
            )
            genetic_generations = 50  # default
            genetic_population = 100  # default
        else:
            col_gen1, col_gen2 = st.columns(2)
            with col_gen1:
                genetic_population = st.select_slider(
                    "Population Size",
                    options=[50, 100, 200, 500],
                    value=100,
                    help="Number of portfolios per generation",
                    disabled=(st.session_state.is_generating and not st.session_state.is_paused)
                )
            with col_gen2:
                genetic_generations = st.select_slider(
                    "Generations",
                    options=[20, 50, 100, 200],
                    value=50,
                    help="Number of evolution cycles",
                    disabled=(st.session_state.is_generating and not st.session_state.is_paused)
                )
            sampling_iterations = 100000  # default

        st.divider()

        # Show current stats if we have portfolios or are generating
        if st.session_state.portfolios or st.session_state.is_generating:
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            with col_stat1:
                if st.session_state.portfolios:
                    best = st.session_state.portfolios[0]
                    st.metric("Best Return/DD", f"{best.return_dd_ratio:.2f}")
            with col_stat2:
                st.metric("Iterations", f"{st.session_state.generation_count}")
            with col_stat3:
                st.metric("Combinations Explored", f"{st.session_state.total_explored:,}")

        # Build buttons - Start/Pause/Resume/Stop style like QuantAnalyzer
        is_running = st.session_state.is_generating and not st.session_state.is_paused
        is_paused = st.session_state.is_generating and st.session_state.is_paused
        is_stopped = not st.session_state.is_generating

        col_start, col_pause, col_stop, col_clear = st.columns([2, 2, 2, 1])

        with col_start:
            if is_stopped:
                start_clicked = st.button(
                    "▶️ Start",
                    type="primary",
                    use_container_width=True,
                    help="Start fresh - clears existing and begins continuous generation"
                )
            elif is_paused:
                start_clicked = st.button(
                    "▶️ Resume",
                    type="primary",
                    use_container_width=True,
                    help="Resume generation from where you paused"
                )
            else:
                start_clicked = st.button(
                    "⏳ Running...",
                    type="primary",
                    use_container_width=True,
                    disabled=True
                )

        with col_pause:
            pause_clicked = st.button(
                "⏸️ Pause",
                use_container_width=True,
                disabled=not is_running,
                help="Pause to review results - you can resume anytime"
            )

        with col_stop:
            stop_clicked = st.button(
                "⏹️ Stop",
                use_container_width=True,
                disabled=is_stopped,
                help="Stop generation completely"
            )

        with col_clear:
            if st.button("🗑️ Clear", use_container_width=True, disabled=is_running):
                st.session_state.portfolios = []
                st.session_state.generation_count = 0
                st.session_state.total_explored = 0
                st.session_state.is_generating = False
                st.session_state.is_paused = False
                st.rerun()

        # Status messages
        if is_running:
            st.info("🔄 **Generating continuously...** Click **Pause** to review results or **Stop** when done.")
        elif is_paused:
            st.warning(f"⏸️ **Paused** after {st.session_state.generation_count} iterations. Review results in the Results tab, then click **Resume** to continue or **Stop** to finish.")
        else:
            st.caption("💡 Like QuantAnalyzer: Click Start and let it run. Pause anytime to review results.")

        # Handle Pause button
        if pause_clicked:
            st.session_state.is_paused = True
            st.rerun()

        # Handle Stop button
        if stop_clicked:
            st.session_state.is_generating = False
            st.session_state.is_paused = False
            if st.session_state.portfolios:
                st.success(f"✅ Stopped after {st.session_state.generation_count} iterations. Found {len(st.session_state.portfolios)} portfolios.")
            st.rerun()

        # Handle Start/Resume button
        if start_clicked:
            if is_stopped:
                # Fresh start
                st.session_state.is_generating = True
                st.session_state.is_paused = False
                st.session_state.generation_count = 0
                st.session_state.total_explored = 0
                st.session_state.portfolios = []
            else:
                # Resume from pause
                st.session_state.is_paused = False
            st.rerun()

        # Run optimization if generating and not paused
        if st.session_state.is_generating and not st.session_state.is_paused:
            filters = {
                'min_sharpe': min_sharpe,
                'min_profit_factor': min_pf,
                'min_trades': min_trades
            }

            settings = {
                'min_strategies': min_strategies,
                'max_strategies': max_strategies,
                'max_per_symbol': max_per_symbol,
                'max_correlation': max_correlation,
                'max_portfolio_equity_dd': max_portfolio_equity_dd,
                'max_portfolio_balance_dd': max_portfolio_balance_dd,
                'rank_by': rank_by,
                'keep_top': keep_top
            }

            locked = st.session_state.locked_indices if mode == "🔄 Add to Locked Strategies" else set()

            progress_bar = st.progress(0)
            status_text = st.empty()

            import time
            start_time = time.time()
            current_iteration = st.session_state.generation_count + 1

            def update_progress(pct):
                elapsed = time.time() - start_time
                progress_bar.progress(min(pct, 1.0))
                status_text.text(f"Iteration {current_iteration} | Sampling... {pct*100:.1f}% | Elapsed: {elapsed:.1f}s")

            status_text.text(f"Iteration {current_iteration} | Building correlation matrix...")

            # Use appropriate search algorithm
            if search_algorithm == "Genetic Search":
                new_portfolios = generate_portfolios_genetic(
                    st.session_state.strategies,
                    locked,
                    filters,
                    settings,
                    update_progress,
                    population_size=genetic_population,
                    generations=genetic_generations
                )
                explored_count = genetic_population * genetic_generations
            else:
                new_portfolios = generate_portfolios(
                    st.session_state.strategies,
                    locked,
                    filters,
                    settings,
                    update_progress,
                    max_iterations=sampling_iterations
                )
                explored_count = sampling_iterations

            # Update counters
            st.session_state.generation_count += 1
            st.session_state.total_explored += explored_count

            # Merge with existing portfolios
            if st.session_state.portfolios:
                all_portfolios = st.session_state.portfolios + new_portfolios

                # Remove duplicates (same strategy_indices)
                seen = set()
                unique_portfolios = []
                for p in all_portfolios:
                    key = tuple(sorted(p.strategy_indices))
                    if key not in seen:
                        seen.add(key)
                        unique_portfolios.append(p)

                # Sort and keep top N
                if settings['rank_by'] == 'Return/DD':
                    unique_portfolios.sort(key=lambda p: p.return_dd_ratio, reverse=True)
                elif settings['rank_by'] == 'Sharpe':
                    unique_portfolios.sort(key=lambda p: p.sharpe, reverse=True)
                elif settings['rank_by'] == 'Net Profit':
                    unique_portfolios.sort(key=lambda p: p.total_profit, reverse=True)
                elif settings['rank_by'] == 'Lowest DD':
                    unique_portfolios.sort(key=lambda p: p.equity_dd, reverse=False)
                elif settings['rank_by'] == 'Lowest Correlation':
                    unique_portfolios.sort(key=lambda p: p.avg_correlation, reverse=False)

                st.session_state.portfolios = unique_portfolios[:keep_top]
            else:
                st.session_state.portfolios = new_portfolios

            elapsed = time.time() - start_time
            progress_bar.progress(1.0)

            if st.session_state.portfolios:
                best = st.session_state.portfolios[0]
                status_text.text(f"✅ Iteration {current_iteration} done ({elapsed:.1f}s) | Best Return/DD: {best.return_dd_ratio:.2f} | Continuing...")
            else:
                status_text.text(f"Iteration {current_iteration} done | No portfolios yet, continuing...")

            # Small delay to allow UI to update, then continue generating
            time.sleep(0.5)
            st.rerun()  # Continue to next iteration

    # ========================================================================
    # TAB 3: RESULTS
    # ========================================================================
    with tab3:
        st.header("Portfolio Results")

        # Save/Load Portfolios section
        col_save, col_load = st.columns(2)

        with col_save:
            with st.expander("💾 Save Portfolios", expanded=False):
                portfolio_save_name = st.text_input(
                    "Portfolio set name",
                    value=f"portfolios_{datetime.now().strftime('%Y%m%d')}",
                    help="Enter a name for this portfolio set",
                    key="portfolio_save_name"
                )

                # Portfolio selection
                if st.session_state.portfolios:
                    save_options = [f"#{i+1} - R/DD: {p.return_dd_ratio:.2f}, Profit: ${p.total_profit:,.0f}"
                                   for i, p in enumerate(st.session_state.portfolios)]

                    save_mode = st.radio("Save mode", ["Save All", "Select Specific"],
                                        horizontal=True, key="save_mode")

                    if save_mode == "Select Specific":
                        selected_to_save = st.multiselect(
                            "Select portfolios to save",
                            options=range(len(st.session_state.portfolios)),
                            default=[0] if st.session_state.portfolios else [],
                            format_func=lambda x: save_options[x],
                            key="portfolios_to_save"
                        )
                        portfolios_to_save = [st.session_state.portfolios[i] for i in selected_to_save]
                    else:
                        portfolios_to_save = st.session_state.portfolios
                        selected_to_save = list(range(len(st.session_state.portfolios)))

                    st.caption(f"Will save {len(portfolios_to_save)} portfolio(s)")

                    if st.button("💾 Save Portfolios", disabled=len(portfolios_to_save) == 0):
                        if portfolios_to_save:
                            try:
                                save_portfolios(portfolio_save_name, portfolios_to_save)
                                st.success(f"Saved {len(portfolios_to_save)} portfolios!")
                            except Exception as e:
                                st.error(f"Error saving: {str(e)}")
                        else:
                            st.warning("No portfolios selected to save")
                else:
                    st.info("No portfolios to save")

        with col_load:
            with st.expander("📂 Load Portfolios", expanded=False):
                saved_portfolio_files = get_saved_portfolios()
                if saved_portfolio_files:
                    selected_portfolio_file = st.selectbox(
                        "Select portfolio set",
                        saved_portfolio_files,
                        help="Choose previously saved portfolios",
                        key="portfolio_load_select"
                    )
                    col_load_btn, col_del_btn = st.columns(2)
                    with col_load_btn:
                        if st.button("📂 Load", key="load_portfolios_btn"):
                            try:
                                loaded = load_portfolios(selected_portfolio_file)
                                st.session_state.portfolios = loaded
                                st.success(f"Loaded {len(loaded)} portfolios!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error loading: {str(e)}")
                    with col_del_btn:
                        if st.button("🗑️ Delete", key="delete_portfolios_btn"):
                            if delete_saved_portfolios(selected_portfolio_file):
                                st.success(f"Deleted '{selected_portfolio_file}'")
                                st.rerun()
                else:
                    st.info("No saved portfolios yet")

        st.divider()

        # Export buttons
        if st.session_state.portfolios and st.session_state.strategies:
            st.subheader("📥 Export Reports")

            # Portfolio selector for export
            export_options = [f"Portfolio #{i+1} (Return/DD: {p.return_dd_ratio:.2f}, Profit: ${p.total_profit:,.0f})"
                            for i, p in enumerate(st.session_state.portfolios)]
            export_idx = st.selectbox(
                "Select Portfolio to Export",
                range(len(export_options)),
                format_func=lambda x: export_options[x],
                key="export_portfolio_selector"
            )

            col_excel, col_html, col_csv = st.columns(3)

            with col_excel:
                try:
                    excel_data = generate_excel_report(
                        st.session_state.portfolios,
                        st.session_state.strategies,
                        selected_idx=export_idx,
                        monte_carlo_result=st.session_state.monte_carlo_result
                    )
                    st.download_button(
                        label="📊 Download Excel Report",
                        data=excel_data,
                        file_name=f"portfolio_{export_idx+1}_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"Excel generation error: {str(e)}")

            with col_html:
                try:
                    html_data = generate_html_report(
                        st.session_state.portfolios,
                        st.session_state.strategies,
                        selected_idx=export_idx,
                        monte_carlo_result=st.session_state.monte_carlo_result
                    )
                    st.download_button(
                        label="🌐 Download HTML Report",
                        data=html_data,
                        file_name=f"portfolio_{export_idx+1}_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                        mime="text/html",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"HTML generation error: {str(e)}")

            with col_csv:
                # Simple CSV export of portfolio rankings
                csv_data = "Rank,Strategies,Net Profit,Equity DD,Balance DD,Return/DD,Avg Correlation,Sharpe,Num Strategies\n"
                for i, p in enumerate(st.session_state.portfolios, start=1):
                    strat_names = '; '.join(p.strategy_names)
                    csv_data += f'{i},"{strat_names}",{p.total_profit:.2f},{p.equity_dd:.2f},{p.balance_dd:.2f},{p.return_dd_ratio:.2f},{p.avg_correlation:.3f},{p.sharpe:.2f},{len(p.strategy_names)}\n'

                st.download_button(
                    label="📄 Download CSV Data",
                    data=csv_data,
                    file_name=f"portfolio_rankings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            st.caption("Excel and HTML reports include charts. CSV is raw data only.")
            st.divider()

        if not st.session_state.portfolios:
            st.info("No portfolios generated yet. Go to Portfolio Builder tab to create portfolios.")
        else:
            # Portfolio list
            st.subheader(f"Top {len(st.session_state.portfolios)} Portfolios")

            for i, portfolio in enumerate(st.session_state.portfolios):
                with st.expander(
                    f"**Portfolio #{i+1}** | Return/DD: {portfolio.return_dd_ratio:.2f} | "
                    f"Profit: ${portfolio.total_profit:,.0f} | Equity DD: ${portfolio.equity_dd:,.0f} | "
                    f"Stability: {portfolio.stability:.2f} | Corr: {portfolio.avg_correlation:.2f}",
                    expanded=(i == 0)
                ):
                    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)

                    with col1:
                        st.metric("Total Profit", f"${portfolio.total_profit:,.2f}")
                    with col2:
                        st.metric("Equity DD", f"${portfolio.equity_dd:,.2f}",
                                 help="Estimated max open/floating drawdown")
                    with col3:
                        st.metric("Balance DD", f"${portfolio.balance_dd:,.2f}",
                                 help="Actual max closed trade drawdown")
                    with col4:
                        st.metric("Return/DD", f"{portfolio.return_dd_ratio:.2f}")
                    with col5:
                        st.metric("Stability", f"{portfolio.stability:.2f}",
                                 help="R² of equity curve (1.0 = perfect)")
                    with col6:
                        st.metric("Symmetry", f"{portfolio.symmetry*100:.0f}%",
                                 help="Profit distribution consistency")
                    with col7:
                        st.metric("Avg Correlation", f"{portfolio.avg_correlation:.3f}")

                    st.markdown("**Strategies in portfolio:**")
                    for idx, name in zip(portfolio.strategy_indices, portfolio.strategy_names):
                        is_locked = idx in st.session_state.locked_indices
                        badge = "🔒 LOCKED" if is_locked else ""
                        strategy = st.session_state.strategies[idx]
                        st.markdown(f"- **{name}** ({strategy.symbol}) - "
                                   f"Profit: ${strategy.metrics.total_net_profit:,.0f}, "
                                   f"Sharpe: {strategy.metrics.sharpe_ratio:.2f} {badge}")

                    st.markdown("**Symbol distribution:**")
                    symbol_str = ", ".join([f"{sym}: {cnt}" for sym, cnt in portfolio.symbols.items()])
                    st.markdown(symbol_str)

    # ========================================================================
    # TAB 4: ANALYSIS
    # ========================================================================
    with tab4:
        st.header("Portfolio Analysis")

        # Select portfolio to analyze
        if st.session_state.portfolios:
            portfolio_options = [f"Portfolio #{i+1} (Return/DD: {p.return_dd_ratio:.2f})"
                               for i, p in enumerate(st.session_state.portfolios)]
            selected_idx = st.selectbox("Select Portfolio", range(len(portfolio_options)),
                                       format_func=lambda x: portfolio_options[x])
            selected_portfolio = st.session_state.portfolios[selected_idx]
        elif st.session_state.selected_portfolio:
            selected_portfolio = st.session_state.selected_portfolio
        else:
            selected_portfolio = None

        if selected_portfolio:
            # Get strategies in portfolio
            portfolio_strategies = [st.session_state.strategies[i]
                                   for i in selected_portfolio.strategy_indices]

            # Metrics overview
            st.subheader("Portfolio Metrics")
            col1, col2, col3, col4, col5, col6 = st.columns(6)

            with col1:
                st.metric("Total Profit", f"${selected_portfolio.total_profit:,.2f}")
            with col2:
                st.metric("Equity DD", f"${selected_portfolio.equity_dd:,.2f}",
                         help="Estimated max open drawdown")
            with col3:
                st.metric("Balance DD", f"${selected_portfolio.balance_dd:,.2f}",
                         help="Actual max closed drawdown")
            with col4:
                st.metric("Return/DD", f"{selected_portfolio.return_dd_ratio:.2f}")
            with col5:
                st.metric("Avg Sharpe", f"{selected_portfolio.sharpe:.2f}")
            with col6:
                st.metric("Avg Correlation", f"{selected_portfolio.avg_correlation:.3f}")

            st.divider()

            # Equity curves
            st.subheader("Equity Curves")
            fig = plot_equity_curves(portfolio_strategies, "Portfolio Equity Curves")
            st.plotly_chart(fig, use_container_width=True)

            # Calculate advanced stats
            adv_stats = calculate_advanced_stats(portfolio_strategies)

            if adv_stats:
                # Advanced Statistics Section
                st.subheader("Advanced Statistics")

                col_a1, col_a2, col_a3, col_a4, col_a5, col_a6 = st.columns(6)
                with col_a1:
                    st.metric("CAGR", f"{adv_stats['cagr']*100:.2f}%",
                             help="Compound Annual Growth Rate")
                with col_a2:
                    st.metric("SQN Score", f"{adv_stats['sqn']:.2f}",
                             help="System Quality Number (>2 good, >3 excellent)")
                with col_a3:
                    st.metric("Stability", f"{adv_stats['stability']:.2f}",
                             help="R² of equity curve (1.0 = perfect straight line)")
                with col_a4:
                    st.metric("Symmetry", f"{adv_stats['symmetry']*100:.1f}%",
                             help="Profit distribution consistency")
                with col_a5:
                    st.metric("Z-Score", f"{adv_stats['z_score']:.2f}",
                             help="Trade dependency measure")
                with col_a6:
                    st.metric("R-Expectancy", f"{adv_stats['r_expectancy']:.2f}R",
                             help="Risk-adjusted expectancy")

                col_b1, col_b2, col_b3, col_b4 = st.columns(4)
                with col_b1:
                    st.metric("Stagnation", f"{adv_stats['stagnation_days']} days",
                             help=f"{adv_stats['stagnation_pct']*100:.1f}% of total period")
                with col_b2:
                    st.metric("Profit Factor", f"{adv_stats['profit_factor']:.2f}")
                with col_b3:
                    st.metric("Avg Trade", f"${adv_stats['avg_trade']:.2f}")
                with col_b4:
                    st.metric("Win Rate", f"{adv_stats['win_rate']*100:.1f}%")

                st.divider()

                # Trade Statistics Section
                st.subheader("Trade Statistics")
                col_t1, col_t2 = st.columns(2)

                with col_t1:
                    st.markdown("**Wins & Losses**")
                    trade_stats = pd.DataFrame({
                        'Metric': ['# of Wins', '# of Losses', 'Gross Profit', 'Gross Loss',
                                  'Avg Win', 'Avg Loss', 'Largest Win', 'Largest Loss'],
                        'Value': [
                            f"{adv_stats['num_wins']}",
                            f"{adv_stats['num_losses']}",
                            f"${adv_stats['gross_profit']:,.2f}",
                            f"${adv_stats['gross_loss']:,.2f}",
                            f"${adv_stats['avg_win']:,.2f}",
                            f"${adv_stats['avg_loss']:,.2f}",
                            f"${adv_stats['largest_win']:,.2f}",
                            f"${adv_stats['largest_loss']:,.2f}"
                        ]
                    })
                    st.dataframe(trade_stats, use_container_width=True, hide_index=True)

                with col_t2:
                    st.markdown("**Consecutive Trades**")
                    consec_stats = pd.DataFrame({
                        'Metric': ['Max Consec Wins', 'Max Consec Losses', 'Total Trades',
                                  'Long Trades', 'Short Trades', 'Long Profit', 'Short Profit'],
                        'Value': [
                            f"{adv_stats['max_consec_wins']}",
                            f"{adv_stats['max_consec_losses']}",
                            f"{adv_stats['total_trades']}",
                            f"{adv_stats['long_trades']}",
                            f"{adv_stats['short_trades']}",
                            f"${adv_stats['long_profit']:,.2f}",
                            f"${adv_stats['short_profit']:,.2f}"
                        ]
                    })
                    st.dataframe(consec_stats, use_container_width=True, hide_index=True)

                st.divider()

                # Long/Short Breakdown
                st.subheader("Long/Short Analysis")
                col_ls1, col_ls2 = st.columns(2)

                with col_ls1:
                    # Trade count pie chart
                    fig_ls_count = go.Figure(data=[go.Pie(
                        labels=['Long Trades', 'Short Trades'],
                        values=[adv_stats['long_trades'], adv_stats['short_trades']],
                        hole=0.4,
                        marker_colors=['#4CAF50', '#FF6B6B']
                    )])
                    fig_ls_count.update_layout(
                        title='Trade Distribution',
                        height=300
                    )
                    st.plotly_chart(fig_ls_count, use_container_width=True)

                with col_ls2:
                    # Profit pie chart
                    long_pct = adv_stats['long_profit'] / (abs(adv_stats['long_profit']) + abs(adv_stats['short_profit'])) * 100 if (adv_stats['long_profit'] + adv_stats['short_profit']) != 0 else 50
                    fig_ls_profit = go.Figure(data=[go.Pie(
                        labels=['Long Profit', 'Short Profit'],
                        values=[max(0, adv_stats['long_profit']), max(0, adv_stats['short_profit'])],
                        hole=0.4,
                        marker_colors=['#4CAF50', '#FF6B6B']
                    )])
                    fig_ls_profit.update_layout(
                        title='Profit Distribution',
                        height=300
                    )
                    st.plotly_chart(fig_ls_profit, use_container_width=True)

                st.divider()

                # P/L by Hour
                st.subheader("P/L by Hour")
                pl_by_hour = adv_stats['pl_by_hour']
                hours = list(range(24))
                hour_profits = [pl_by_hour.get(h, 0) for h in hours]
                colors_hour = ['#4CAF50' if p > 0 else '#FF6B6B' for p in hour_profits]

                fig_hour = go.Figure(data=[go.Bar(
                    x=hours,
                    y=hour_profits,
                    marker_color=colors_hour,
                    text=[f"${p:,.0f}" for p in hour_profits],
                    textposition='outside'
                )])
                fig_hour.update_layout(
                    title='Profit/Loss by Hour of Day',
                    xaxis_title='Hour (0-23)',
                    yaxis_title='Profit/Loss ($)',
                    height=350,
                    xaxis=dict(tickmode='linear', tick0=0, dtick=1)
                )
                st.plotly_chart(fig_hour, use_container_width=True)

                # P/L by Day of Week
                st.subheader("P/L by Day of Week")
                pl_by_day = adv_stats['pl_by_day']
                day_order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
                day_profits = [pl_by_day.get(d, 0) for d in day_order]
                colors_day = ['#4CAF50' if p > 0 else '#FF6B6B' for p in day_profits]

                fig_day = go.Figure(data=[go.Bar(
                    x=day_order,
                    y=day_profits,
                    marker_color=colors_day,
                    text=[f"${p:,.0f}" for p in day_profits],
                    textposition='outside'
                )])
                fig_day.update_layout(
                    title='Profit/Loss by Day of Week',
                    xaxis_title='Day',
                    yaxis_title='Profit/Loss ($)',
                    height=350
                )
                st.plotly_chart(fig_day, use_container_width=True)

                st.divider()

                # Detailed Trade List
                st.subheader("Trade List")
                trades_df = adv_stats.get('trades_df')
                if trades_df is not None and len(trades_df) > 0:
                    with st.expander(f"📋 View All {len(trades_df)} Trades", expanded=False):
                        display_trades = trades_df[['datetime', 'strategy', 'type', 'profit', 'cumulative_profit', 'balance']].copy()
                        display_trades['datetime'] = display_trades['datetime'].dt.strftime('%Y-%m-%d %H:%M')
                        display_trades.columns = ['Date/Time', 'Strategy', 'Type', 'Profit', 'Cumulative P/L', 'Balance']
                        display_trades['Profit'] = display_trades['Profit'].apply(lambda x: f"${x:,.2f}")
                        display_trades['Cumulative P/L'] = display_trades['Cumulative P/L'].apply(lambda x: f"${x:,.2f}")
                        display_trades['Balance'] = display_trades['Balance'].apply(lambda x: f"${x:,.2f}")
                        st.dataframe(display_trades, use_container_width=True, height=400)

                st.divider()

            # Monthly P/L Analysis
            st.subheader("Monthly P/L Analysis")
            equity_df = build_equity_curve(portfolio_strategies)

            if not equity_df.empty:
                # Calculate monthly returns
                equity_df['month'] = equity_df['datetime'].dt.to_period('M')
                monthly_pnl = equity_df.groupby('month')['profit'].sum().reset_index()
                monthly_pnl['month_str'] = monthly_pnl['month'].astype(str)
                monthly_pnl['year'] = monthly_pnl['month'].dt.year
                monthly_pnl['month_num'] = monthly_pnl['month'].dt.month

                # Summary stats
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                with col_m1:
                    st.metric("Avg Monthly P/L", f"${monthly_pnl['profit'].mean():,.2f}")
                with col_m2:
                    st.metric("Best Month", f"${monthly_pnl['profit'].max():,.2f}")
                with col_m3:
                    st.metric("Worst Month", f"${monthly_pnl['profit'].min():,.2f}")
                with col_m4:
                    win_months = (monthly_pnl['profit'] > 0).sum()
                    total_months = len(monthly_pnl)
                    st.metric("Win Rate", f"{win_months}/{total_months} ({100*win_months/total_months:.0f}%)")

                # Monthly heatmap
                if len(monthly_pnl) > 1:
                    pivot = monthly_pnl.pivot_table(values='profit', index='year', columns='month_num', aggfunc='sum')
                    pivot = pivot.reindex(columns=range(1, 13), fill_value=0)

                    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

                    # Add yearly totals
                    pivot['Total'] = pivot.sum(axis=1)

                    fig_heatmap = go.Figure(data=go.Heatmap(
                        z=pivot.iloc[:, :-1].values,
                        x=month_labels,
                        y=pivot.index.astype(str),
                        colorscale=[[0, '#FF6B6B'], [0.5, '#FFFFFF'], [1, '#4CAF50']],
                        zmid=0,
                        text=np.round(pivot.iloc[:, :-1].values, 0),
                        texttemplate='$%{text:.0f}',
                        textfont={"size": 10},
                        hoverongaps=False,
                        colorbar=dict(title="P/L ($)")
                    ))
                    fig_heatmap.update_layout(
                        title='Monthly Returns Heatmap',
                        xaxis_title='Month',
                        yaxis_title='Year',
                        height=max(200, len(pivot) * 50 + 100)
                    )
                    st.plotly_chart(fig_heatmap, use_container_width=True)

                    # Monthly P/L table
                    with st.expander("📊 Monthly P/L Table", expanded=False):
                        # Format the pivot table for display
                        display_pivot = pivot.copy()
                        display_pivot.columns = month_labels + ['Total']
                        display_pivot = display_pivot.apply(lambda col: col.map(lambda x: f"${x:,.0f}" if x != 0 else "-"))
                        st.dataframe(display_pivot, use_container_width=True)

                    # Monthly bar chart
                    fig_bar = go.Figure()
                    colors = ['#4CAF50' if x > 0 else '#FF6B6B' for x in monthly_pnl['profit']]
                    fig_bar.add_trace(go.Bar(
                        x=monthly_pnl['month_str'],
                        y=monthly_pnl['profit'],
                        marker_color=colors,
                        text=[f"${x:,.0f}" for x in monthly_pnl['profit']],
                        textposition='outside'
                    ))
                    fig_bar.update_layout(
                        title='Monthly P/L Over Time',
                        xaxis_title='Month',
                        yaxis_title='Profit/Loss ($)',
                        height=350,
                        showlegend=False
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.info("No trade data available for monthly analysis.")

            st.divider()

            # Correlation heatmap
            st.subheader("Correlation Matrix")
            fig = plot_correlation_heatmap(portfolio_strategies)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

            # Individual strategy metrics
            st.subheader("Individual Strategy Metrics")
            metrics_data = []
            for s in portfolio_strategies:
                is_locked = st.session_state.strategies.index(s) in st.session_state.locked_indices
                metrics_data.append({
                    'Status': '🔒' if is_locked else '',
                    'Name': s.name,
                    'Symbol': s.symbol,
                    'Net Profit': f"${s.metrics.total_net_profit:,.2f}",
                    'DD': f"${s.metrics.equity_dd_maximal:,.2f}",
                    'Sharpe': f"{s.metrics.sharpe_ratio:.2f}",
                    'PF': f"{s.metrics.profit_factor:.2f}",
                    'Win Rate': f"{s.metrics.win_rate:.1f}%",
                    'Trades': s.metrics.total_trades
                })

            st.dataframe(pd.DataFrame(metrics_data), use_container_width=True)

            st.divider()

            # ================================================================
            # WHAT IF ANALYSIS
            # ================================================================
            st.subheader("🔮 What If Analysis")
            st.markdown("""
            Filter trades to see how your portfolio would perform under different conditions.
            Exclude specific times, days, or trade types to optimize your strategy.
            """)

            # Get trades data for filtering
            if adv_stats and 'trades_df' in adv_stats and adv_stats['trades_df'] is not None:
                trades_df = adv_stats['trades_df'].copy()

                if len(trades_df) > 0:
                    # Filter controls in columns
                    col_wf1, col_wf2 = st.columns(2)

                    with col_wf1:
                        # Hour filter
                        all_hours = list(range(24))
                        excluded_hours = st.multiselect(
                            "Exclude Hours",
                            options=all_hours,
                            default=[],
                            help="Select hours to exclude from analysis (0-23)",
                            key="whatif_hours"
                        )

                        # Day of week filter
                        day_options = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                        excluded_days = st.multiselect(
                            "Exclude Days",
                            options=day_options,
                            default=[],
                            help="Select days to exclude from analysis",
                            key="whatif_days"
                        )

                    with col_wf2:
                        # Trade direction filter
                        direction_options = ['Long', 'Short']
                        included_directions = st.multiselect(
                            "Include Trade Types",
                            options=direction_options,
                            default=direction_options,
                            help="Select which trade types to include",
                            key="whatif_direction"
                        )

                        # Month filter
                        trades_df['month'] = trades_df['datetime'].dt.to_period('M')
                        available_months = sorted(trades_df['month'].unique().astype(str).tolist())
                        excluded_months = st.multiselect(
                            "Exclude Months",
                            options=available_months,
                            default=[],
                            help="Select specific months to exclude",
                            key="whatif_months"
                        )

                    # Apply filters
                    filtered_trades = trades_df.copy()

                    # Filter by hour
                    if excluded_hours:
                        filtered_trades = filtered_trades[~filtered_trades['hour'].isin(excluded_hours)]

                    # Filter by day
                    if excluded_days:
                        day_map = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
                                  'Friday': 4, 'Saturday': 5, 'Sunday': 6}
                        excluded_day_nums = [day_map[d] for d in excluded_days]
                        filtered_trades = filtered_trades[~filtered_trades['day_of_week'].isin(excluded_day_nums)]

                    # Filter by direction
                    if included_directions:
                        direction_vals = []
                        if 'Long' in included_directions:
                            direction_vals.append('long')
                        if 'Short' in included_directions:
                            direction_vals.append('short')
                        filtered_trades = filtered_trades[filtered_trades['type'].isin(direction_vals)]

                    # Filter by month
                    if excluded_months:
                        filtered_trades = filtered_trades[~filtered_trades['month'].astype(str).isin(excluded_months)]

                    # Calculate What If results
                    if len(filtered_trades) > 0:
                        # Recalculate metrics
                        wf_total_profit = filtered_trades['profit'].sum()
                        wf_num_trades = len(filtered_trades)
                        wf_wins = (filtered_trades['profit'] > 0).sum()
                        wf_losses = (filtered_trades['profit'] <= 0).sum()
                        wf_win_rate = wf_wins / wf_num_trades if wf_num_trades > 0 else 0
                        wf_avg_trade = filtered_trades['profit'].mean()
                        wf_gross_profit = filtered_trades[filtered_trades['profit'] > 0]['profit'].sum()
                        wf_gross_loss = abs(filtered_trades[filtered_trades['profit'] <= 0]['profit'].sum())
                        wf_profit_factor = wf_gross_profit / wf_gross_loss if wf_gross_loss > 0 else float('inf')

                        # Calculate max DD from filtered trades
                        filtered_trades = filtered_trades.sort_values('datetime')
                        filtered_trades['cumulative'] = filtered_trades['profit'].cumsum()
                        filtered_trades['running_max'] = filtered_trades['cumulative'].cummax()
                        filtered_trades['drawdown'] = filtered_trades['running_max'] - filtered_trades['cumulative']
                        wf_max_dd = filtered_trades['drawdown'].max()
                        wf_return_dd = wf_total_profit / wf_max_dd if wf_max_dd > 0 else float('inf')

                        # Display comparison
                        st.markdown("### Original vs What If Comparison")

                        # Calculate original stats
                        orig_total_profit = trades_df['profit'].sum()
                        orig_num_trades = len(trades_df)
                        orig_win_rate = adv_stats['win_rate']
                        orig_profit_factor = adv_stats['profit_factor']

                        # Create comparison dataframe
                        comparison_df = pd.DataFrame({
                            'Metric': ['Total Profit', 'Number of Trades', 'Win Rate', 'Profit Factor',
                                      'Avg Trade', 'Max Drawdown', 'Return/DD'],
                            'Original': [
                                f"${orig_total_profit:,.2f}",
                                f"{orig_num_trades}",
                                f"{orig_win_rate*100:.1f}%",
                                f"{orig_profit_factor:.2f}",
                                f"${adv_stats['avg_trade']:.2f}",
                                f"${selected_portfolio.equity_dd:,.2f}",
                                f"{selected_portfolio.return_dd_ratio:.2f}"
                            ],
                            'What If': [
                                f"${wf_total_profit:,.2f}",
                                f"{wf_num_trades}",
                                f"{wf_win_rate*100:.1f}%",
                                f"{wf_profit_factor:.2f}",
                                f"${wf_avg_trade:.2f}",
                                f"${wf_max_dd:,.2f}",
                                f"{wf_return_dd:.2f}"
                            ],
                            'Change': [
                                f"{((wf_total_profit - orig_total_profit) / abs(orig_total_profit) * 100) if orig_total_profit != 0 else 0:+.1f}%",
                                f"{wf_num_trades - orig_num_trades:+d} ({(wf_num_trades - orig_num_trades)/orig_num_trades*100:+.1f}%)",
                                f"{(wf_win_rate - orig_win_rate)*100:+.1f}pp",
                                f"{wf_profit_factor - orig_profit_factor:+.2f}",
                                f"${wf_avg_trade - adv_stats['avg_trade']:+.2f}",
                                f"{((wf_max_dd - selected_portfolio.equity_dd) / selected_portfolio.equity_dd * 100) if selected_portfolio.equity_dd > 0 else 0:+.1f}%",
                                f"{wf_return_dd - selected_portfolio.return_dd_ratio:+.2f}"
                            ]
                        })

                        col_comp1, col_comp2 = st.columns([2, 1])

                        with col_comp1:
                            st.dataframe(comparison_df, use_container_width=True, hide_index=True)

                        with col_comp2:
                            # Summary insight
                            profit_diff = wf_total_profit - orig_total_profit
                            if profit_diff > 0:
                                st.success(f"📈 Profit increased by ${profit_diff:,.2f}")
                            elif profit_diff < 0:
                                st.error(f"📉 Profit decreased by ${abs(profit_diff):,.2f}")
                            else:
                                st.info("No change in profit")

                            trades_removed = orig_num_trades - wf_num_trades
                            st.info(f"🔢 {trades_removed} trades filtered out ({trades_removed/orig_num_trades*100:.1f}%)")

                        # What If Equity Curve
                        st.markdown("### What If Equity Curve")
                        fig_wf = go.Figure()

                        # Original equity curve
                        orig_equity = trades_df.sort_values('datetime').copy()
                        orig_equity['cumulative'] = orig_equity['profit'].cumsum()
                        fig_wf.add_trace(go.Scatter(
                            x=orig_equity['datetime'],
                            y=orig_equity['cumulative'],
                            mode='lines',
                            name='Original',
                            line=dict(color='#888888', width=1, dash='dash')
                        ))

                        # What If equity curve
                        fig_wf.add_trace(go.Scatter(
                            x=filtered_trades['datetime'],
                            y=filtered_trades['cumulative'],
                            mode='lines',
                            name='What If',
                            line=dict(color='#4CAF50', width=2)
                        ))

                        fig_wf.update_layout(
                            title='Equity Curve Comparison',
                            xaxis_title='Date',
                            yaxis_title='Cumulative P/L ($)',
                            height=400,
                            hovermode='x unified'
                        )
                        st.plotly_chart(fig_wf, use_container_width=True)

                    else:
                        st.warning("⚠️ No trades match the selected filters. Adjust your criteria.")
                else:
                    st.info("No trade data available for What If analysis.")
            else:
                st.info("Calculate advanced stats above to enable What If analysis.")

        else:
            st.info("Select a portfolio from the Results tab or generate portfolios first.")

            # Show all strategies equity curves
            if st.session_state.strategies:
                st.subheader("All Strategies Equity Curves")
                fig = plot_equity_curves(st.session_state.strategies[:10], "All Strategies (first 10)")
                st.plotly_chart(fig, use_container_width=True)

    # ========================================================================
    # TAB 5: MONTE CARLO
    # ========================================================================
    with tab5:
        st.header("🎲 Monte Carlo Portfolio Analysis")
        st.markdown("""
        Monte Carlo simulation stress-tests your portfolio by randomly shuffling trades
        to see the range of possible outcomes. This helps understand:
        - **Confidence intervals** for maximum drawdown
        - **Probability of ruin** (hitting certain loss thresholds)
        - **Expected return range** (best/worst/median outcomes)
        """)

        # Select portfolio or strategies for analysis
        st.subheader("Select Strategies for Analysis")

        analysis_mode = st.radio(
            "Analysis Mode",
            ["Use Selected Portfolio", "Use Locked Strategies", "Use All Strategies"],
            horizontal=True
        )

        if analysis_mode == "Use Selected Portfolio" and st.session_state.portfolios:
            portfolio_options = [f"Portfolio #{i+1} (Return/DD: {p.return_dd_ratio:.2f})"
                               for i, p in enumerate(st.session_state.portfolios)]
            mc_portfolio_idx = st.selectbox("Select Portfolio", range(len(portfolio_options)),
                                           format_func=lambda x: portfolio_options[x],
                                           key="mc_portfolio_select")
            mc_strategies = [st.session_state.strategies[i]
                           for i in st.session_state.portfolios[mc_portfolio_idx].strategy_indices]
        elif analysis_mode == "Use Locked Strategies" and st.session_state.locked_indices:
            mc_strategies = [st.session_state.strategies[i] for i in st.session_state.locked_indices]
        else:
            mc_strategies = st.session_state.strategies

        st.info(f"**{len(mc_strategies)} strategies selected** with "
               f"{sum(len(s.trades) for s in mc_strategies)} total trades")

        st.divider()

        # Monte Carlo Settings
        st.subheader("Simulation Settings")

        col1, col2, col3 = st.columns(3)

        with col1:
            num_simulations = st.select_slider(
                "Number of Simulations",
                options=[100, 500, 1000, 2000, 5000, 10000],
                value=1000,
                help="More simulations = more accurate results but slower"
            )

        with col2:
            initial_balance = st.number_input(
                "Initial Balance ($)",
                min_value=1000.0,
                value=10000.0,
                step=1000.0
            )

        with col3:
            sim_method = st.selectbox(
                "Simulation Method",
                ["shuffle", "bootstrap"],
                help="Shuffle: reorder trades. Bootstrap: sample with replacement"
            )

        col1, col2 = st.columns(2)

        with col1:
            use_parallel = st.checkbox("Use Parallel Processing", value=True,
                                      help="Use multiple CPU cores for faster simulation")

        with col2:
            import multiprocessing
            usable_cores = min(multiprocessing.cpu_count(), 61)  # Windows pool cap
            max_workers = st.slider(
                "CPU Cores",
                1, usable_cores,
                usable_cores,
                help=f"Available: {multiprocessing.cpu_count()} cores (max 61 usable per pool)"
            ) if use_parallel else 1

        st.divider()

        # Run Monte Carlo
        if st.button("🎲 Run Monte Carlo Simulation", type="primary", use_container_width=True):
            if not mc_strategies:
                st.error("No strategies selected for analysis")
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()

                status_text.text(f"Running {num_simulations} simulations using {max_workers} CPU cores...")

                import time
                start_time = time.time()

                try:
                    result = run_portfolio_monte_carlo(
                        mc_strategies,
                        num_simulations=num_simulations,
                        initial_balance=initial_balance,
                        method=sim_method,
                        parallel=use_parallel,
                        max_workers=max_workers if use_parallel else 1
                    )

                    elapsed = time.time() - start_time
                    st.session_state.monte_carlo_result = result

                    progress_bar.progress(1.0)
                    status_text.text(f"✅ Completed {num_simulations} simulations in {elapsed:.2f}s "
                                   f"({num_simulations/elapsed:.0f} sims/sec)")

                except Exception as e:
                    st.error(f"Simulation error: {str(e)}")
                    progress_bar.empty()
                    status_text.empty()

        # Display Results
        if st.session_state.monte_carlo_result:
            result = st.session_state.monte_carlo_result

            st.divider()
            st.subheader("📊 Simulation Results")

            # Key metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Expected Final Equity",
                    f"${result.final_equity_mean:,.0f}",
                    delta=f"±${result.final_equity_std:,.0f}"
                )

            with col2:
                st.metric(
                    "Median Final Equity",
                    f"${result.final_equity_median:,.0f}"
                )

            with col3:
                st.metric(
                    "Expected Max DD",
                    f"{result.max_dd_mean:.1f}%",
                    delta=f"±{result.max_dd_std:.1f}%"
                )

            with col4:
                st.metric(
                    "95% DD Confidence",
                    f"{result.dd_95_ci:.1f}%",
                    help="95% chance max DD stays below this"
                )

            # Probability of Ruin
            st.subheader("⚠️ Probability of Ruin")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                prob_10 = result.prob_ruin_10pct * 100
                st.metric("P(DD ≥ 10%)", f"{prob_10:.1f}%",
                         delta="Low Risk" if prob_10 < 20 else "Medium" if prob_10 < 50 else "High Risk",
                         delta_color="normal" if prob_10 < 20 else "off" if prob_10 < 50 else "inverse")

            with col2:
                prob_20 = result.prob_ruin_20pct * 100
                st.metric("P(DD ≥ 20%)", f"{prob_20:.1f}%",
                         delta="Low Risk" if prob_20 < 10 else "Medium" if prob_20 < 30 else "High Risk",
                         delta_color="normal" if prob_20 < 10 else "off" if prob_20 < 30 else "inverse")

            with col3:
                prob_30 = result.prob_ruin_30pct * 100
                st.metric("P(DD ≥ 30%)", f"{prob_30:.1f}%",
                         delta="Low Risk" if prob_30 < 5 else "Medium" if prob_30 < 15 else "High Risk",
                         delta_color="normal" if prob_30 < 5 else "off" if prob_30 < 15 else "inverse")

            with col4:
                prob_50 = result.prob_ruin_50pct * 100
                st.metric("P(DD ≥ 50%)", f"{prob_50:.1f}%",
                         delta="Low Risk" if prob_50 < 1 else "Medium" if prob_50 < 5 else "High Risk",
                         delta_color="normal" if prob_50 < 1 else "off" if prob_50 < 5 else "inverse")

            # Charts
            st.subheader("📈 Visualizations")

            chart_col1, chart_col2 = st.columns(2)

            with chart_col1:
                # Equity Fan Chart
                if result.percentile_curves and 50 in result.percentile_curves:
                    fig = go.Figure()

                    x_axis = list(range(len(result.percentile_curves[50])))

                    # Add confidence bands
                    if 5 in result.percentile_curves and 95 in result.percentile_curves:
                        fig.add_trace(go.Scatter(
                            x=x_axis + x_axis[::-1],
                            y=result.percentile_curves[95] + result.percentile_curves[5][::-1],
                            fill='toself',
                            fillcolor='rgba(59, 130, 246, 0.2)',
                            line=dict(color='rgba(255,255,255,0)'),
                            name='5th-95th Percentile',
                            showlegend=True
                        ))

                    if 25 in result.percentile_curves and 75 in result.percentile_curves:
                        fig.add_trace(go.Scatter(
                            x=x_axis + x_axis[::-1],
                            y=result.percentile_curves[75] + result.percentile_curves[25][::-1],
                            fill='toself',
                            fillcolor='rgba(34, 197, 94, 0.3)',
                            line=dict(color='rgba(255,255,255,0)'),
                            name='25th-75th Percentile',
                            showlegend=True
                        ))

                    # Median line
                    fig.add_trace(go.Scatter(
                        x=x_axis,
                        y=result.percentile_curves[50],
                        mode='lines',
                        name='Median (50th)',
                        line=dict(color='#22c55e', width=2)
                    ))

                    fig.update_layout(
                        title="Equity Curve Fan Chart",
                        xaxis_title="Trade Sequence",
                        yaxis_title="Equity ($)",
                        template="plotly_dark",
                        height=400
                    )

                    st.plotly_chart(fig, use_container_width=True)

            with chart_col2:
                # Max Drawdown Distribution
                dd_percentiles = list(result.max_dd_percentiles.values())
                dd_labels = [f"{p}th" for p in result.max_dd_percentiles.keys()]

                fig = go.Figure()

                fig.add_trace(go.Bar(
                    x=dd_labels,
                    y=dd_percentiles,
                    marker_color=['#22c55e' if v < 10 else '#f59e0b' if v < 20 else '#ef4444'
                                 for v in dd_percentiles],
                    text=[f"{v:.1f}%" for v in dd_percentiles],
                    textposition='outside'
                ))

                fig.update_layout(
                    title="Max Drawdown by Percentile",
                    xaxis_title="Percentile",
                    yaxis_title="Max Drawdown (%)",
                    template="plotly_dark",
                    height=400
                )

                st.plotly_chart(fig, use_container_width=True)

            # Final Equity Distribution
            st.subheader("Final Equity Distribution")

            eq_percentiles = result.final_equity_percentiles
            fig = go.Figure()

            fig.add_trace(go.Bar(
                x=[f"{p}th" for p in eq_percentiles.keys()],
                y=list(eq_percentiles.values()),
                marker_color='#3b82f6',
                text=[f"${v:,.0f}" for v in eq_percentiles.values()],
                textposition='outside'
            ))

            fig.update_layout(
                title="Final Equity by Percentile",
                xaxis_title="Percentile",
                yaxis_title="Final Equity ($)",
                template="plotly_dark",
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

            # Summary Table
            st.subheader("📋 Summary Statistics")

            summary_data = {
                "Metric": [
                    "Simulations Run",
                    "Final Equity (Mean)",
                    "Final Equity (Median)",
                    "Final Equity (Min)",
                    "Final Equity (Max)",
                    "Max DD (Mean)",
                    "Max DD (95% CI)",
                    "Max DD (99% CI)",
                    "P(DD ≥ 10%)",
                    "P(DD ≥ 20%)",
                    "P(DD ≥ 30%)",
                ],
                "Value": [
                    f"{result.num_simulations:,}",
                    f"${result.final_equity_mean:,.2f}",
                    f"${result.final_equity_median:,.2f}",
                    f"${result.final_equity_min:,.2f}",
                    f"${result.final_equity_max:,.2f}",
                    f"{result.max_dd_mean:.2f}%",
                    f"{result.dd_95_ci:.2f}%",
                    f"{result.dd_99_ci:.2f}%",
                    f"{result.prob_ruin_10pct*100:.1f}%",
                    f"{result.prob_ruin_20pct*100:.1f}%",
                    f"{result.prob_ruin_30pct*100:.1f}%",
                ]
            }

            st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()

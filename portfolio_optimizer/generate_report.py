"""
Static HTML Report Generator

Generates a standalone HTML report with all portfolio analysis,
including Monte Carlo simulations, that can be opened in any browser.
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import json

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from mt5_parser import MT5Parser
from correlation import CorrelationEngine
from optimizer import PortfolioOptimizer, OptimizationConfig
from monte_carlo import run_portfolio_monte_carlo

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np


def generate_report(
    data_dir: str = "data/candidates",
    output_file: str = "portfolio_report.html",
    num_monte_carlo: int = 1000,
    initial_balance: float = 10000.0
):
    """Generate a complete HTML report."""

    print("=" * 60)
    print("PORTFOLIO OPTIMIZER - STATIC REPORT GENERATOR")
    print("=" * 60)

    # Parse strategies
    print(f"\n[1/5] Parsing strategies from {data_dir}...")
    parser = MT5Parser()
    strategies = parser.parse_directory(data_dir, parallel=True)
    print(f"      Loaded {len(strategies)} strategies")

    if not strategies:
        print("ERROR: No strategies found!")
        return

    # Build correlation matrix
    print("\n[2/5] Building correlation matrix...")
    engine = CorrelationEngine(strategies)
    corr_matrix, strategy_names = engine.build_correlation_matrix()
    print(f"      Matrix size: {len(strategy_names)} x {len(strategy_names)}")

    # Run Monte Carlo
    print(f"\n[3/5] Running {num_monte_carlo} Monte Carlo simulations...")
    mc_result = run_portfolio_monte_carlo(
        strategies,
        num_simulations=num_monte_carlo,
        initial_balance=initial_balance,
        method="shuffle",
        parallel=True
    )
    print(f"      Complete!")

    # Generate visualizations
    print("\n[4/5] Generating visualizations...")

    # Equity curves
    equity_fig = create_equity_curves(strategies)

    # Correlation heatmap
    corr_fig = create_correlation_heatmap(corr_matrix, strategy_names)

    # Monte Carlo fan chart
    mc_fan_fig = create_mc_fan_chart(mc_result)

    # Monte Carlo drawdown distribution
    mc_dd_fig = create_mc_dd_chart(mc_result)

    # Strategy comparison table
    strategy_table = create_strategy_table(strategies)

    # Generate HTML
    print("\n[5/5] Generating HTML report...")

    html = generate_html(
        strategies=strategies,
        mc_result=mc_result,
        equity_fig=equity_fig,
        corr_fig=corr_fig,
        mc_fan_fig=mc_fan_fig,
        mc_dd_fig=mc_dd_fig,
        strategy_table=strategy_table,
        initial_balance=initial_balance
    )

    # Write to file
    output_path = Path(output_file)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"\n{'=' * 60}")
    print(f"REPORT GENERATED: {output_path.absolute()}")
    print(f"{'=' * 60}")
    print(f"\nOpen this file in your browser to view the report.")

    return str(output_path.absolute())


def create_equity_curves(strategies):
    """Create equity curves chart."""
    fig = go.Figure()

    colors = [
        '#3b82f6', '#22c55e', '#f59e0b', '#ef4444', '#8b5cf6',
        '#ec4899', '#06b6d4', '#84cc16', '#f97316', '#6366f1'
    ]

    for i, strategy in enumerate(strategies[:10]):  # Limit to 10 for readability
        daily_returns = strategy.get_daily_returns()
        if not daily_returns:
            continue

        dates = sorted(daily_returns.keys())
        cumulative = []
        running = strategy.metrics.initial_deposit or 10000

        for date in dates:
            running += daily_returns[date]
            cumulative.append(running)

        fig.add_trace(go.Scatter(
            x=dates,
            y=cumulative,
            name=strategy.name[:25] + '...' if len(strategy.name) > 25 else strategy.name,
            line=dict(color=colors[i % len(colors)], width=2)
        ))

    fig.update_layout(
        title="Strategy Equity Curves",
        xaxis_title="Date",
        yaxis_title="Equity ($)",
        template="plotly_dark",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig.to_html(full_html=False, include_plotlyjs=False)


def create_correlation_heatmap(matrix, names):
    """Create correlation heatmap."""
    display_names = [n[:15] + '..' if len(n) > 15 else n for n in names]

    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=display_names,
        y=display_names,
        colorscale=[
            [0, '#3b82f6'],
            [0.35, '#22c55e'],
            [0.5, '#84cc16'],
            [0.65, '#f59e0b'],
            [1, '#ef4444']
        ],
        zmid=0,
        text=np.round(matrix, 2),
        texttemplate="%{text}",
        textfont={"size": 9}
    ))

    fig.update_layout(
        title="Strategy Correlation Matrix",
        template="plotly_dark",
        height=500,
        xaxis=dict(tickangle=45)
    )

    return fig.to_html(full_html=False, include_plotlyjs=False)


def create_mc_fan_chart(mc_result):
    """Create Monte Carlo fan chart."""
    fig = go.Figure()

    if mc_result.percentile_curves and 50 in mc_result.percentile_curves:
        x_axis = list(range(len(mc_result.percentile_curves[50])))

        # 5-95 percentile band
        if 5 in mc_result.percentile_curves and 95 in mc_result.percentile_curves:
            fig.add_trace(go.Scatter(
                x=x_axis + x_axis[::-1],
                y=mc_result.percentile_curves[95] + mc_result.percentile_curves[5][::-1],
                fill='toself',
                fillcolor='rgba(59, 130, 246, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='5th-95th Percentile'
            ))

        # 25-75 percentile band
        if 25 in mc_result.percentile_curves and 75 in mc_result.percentile_curves:
            fig.add_trace(go.Scatter(
                x=x_axis + x_axis[::-1],
                y=mc_result.percentile_curves[75] + mc_result.percentile_curves[25][::-1],
                fill='toself',
                fillcolor='rgba(34, 197, 94, 0.3)',
                line=dict(color='rgba(255,255,255,0)'),
                name='25th-75th Percentile'
            ))

        # Median line
        fig.add_trace(go.Scatter(
            x=x_axis,
            y=mc_result.percentile_curves[50],
            mode='lines',
            name='Median (50th)',
            line=dict(color='#22c55e', width=2)
        ))

    fig.update_layout(
        title="Monte Carlo Equity Fan Chart",
        xaxis_title="Trade Sequence",
        yaxis_title="Equity ($)",
        template="plotly_dark",
        height=400
    )

    return fig.to_html(full_html=False, include_plotlyjs=False)


def create_mc_dd_chart(mc_result):
    """Create drawdown distribution chart."""
    dd_percentiles = list(mc_result.max_dd_percentiles.values())
    dd_labels = [f"{p}th" for p in mc_result.max_dd_percentiles.keys()]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=dd_labels,
        y=dd_percentiles,
        marker_color=['#22c55e' if v < 10 else '#f59e0b' if v < 20 else '#ef4444' for v in dd_percentiles],
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

    return fig.to_html(full_html=False, include_plotlyjs=False)


def create_strategy_table(strategies):
    """Create HTML table of strategy metrics."""
    rows = []
    for s in sorted(strategies, key=lambda x: x.metrics.total_net_profit, reverse=True):
        m = s.metrics
        return_dd = m.total_net_profit / m.equity_dd_maximal if m.equity_dd_maximal > 0 else 0
        rows.append(f"""
        <tr>
            <td>{s.name[:30]}{'...' if len(s.name) > 30 else ''}</td>
            <td>{s.symbol}</td>
            <td>${m.total_net_profit:,.2f}</td>
            <td>${m.equity_dd_maximal:,.2f}</td>
            <td>{m.equity_dd_maximal_pct:.2f}%</td>
            <td>{m.sharpe_ratio:.2f}</td>
            <td>{m.profit_factor:.2f}</td>
            <td>{return_dd:.2f}</td>
            <td>{m.win_rate:.1f}%</td>
            <td>{m.total_trades}</td>
        </tr>
        """)

    return "\n".join(rows)


def generate_html(strategies, mc_result, equity_fig, corr_fig, mc_fan_fig, mc_dd_fig, strategy_table, initial_balance):
    """Generate complete HTML report."""

    total_profit = sum(s.metrics.total_net_profit for s in strategies)
    avg_sharpe = np.mean([s.metrics.sharpe_ratio for s in strategies])
    total_trades = sum(s.metrics.total_trades for s in strategies)

    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Portfolio Optimizer Report</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0f172a;
            color: #e2e8f0;
            margin: 0;
            padding: 20px;
            line-height: 1.6;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        h1 {{ color: #3b82f6; border-bottom: 2px solid #3b82f6; padding-bottom: 10px; }}
        h2 {{ color: #22c55e; margin-top: 40px; }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: #1e293b;
            border-radius: 8px;
            padding: 20px;
            text-align: center;
            border: 1px solid #334155;
        }}
        .metric-value {{ font-size: 28px; font-weight: bold; color: #3b82f6; }}
        .metric-label {{ color: #94a3b8; font-size: 14px; margin-top: 5px; }}
        .chart-container {{ margin: 30px 0; background: #1e293b; border-radius: 8px; padding: 20px; }}
        .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: #1e293b;
            border-radius: 8px;
            overflow: hidden;
        }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #334155; }}
        th {{ background: #334155; color: #e2e8f0; }}
        tr:hover {{ background: #334155; }}
        .risk-low {{ color: #22c55e; }}
        .risk-medium {{ color: #f59e0b; }}
        .risk-high {{ color: #ef4444; }}
        .timestamp {{ color: #64748b; font-size: 12px; margin-top: 40px; text-align: center; }}
        @media (max-width: 768px) {{
            .grid-2 {{ grid-template-columns: 1fr; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Portfolio Optimizer Report</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

        <h2>📋 Portfolio Summary</h2>
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value">{len(strategies)}</div>
                <div class="metric-label">Total Strategies</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">${total_profit:,.0f}</div>
                <div class="metric-label">Combined Net Profit</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{avg_sharpe:.2f}</div>
                <div class="metric-label">Average Sharpe Ratio</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{total_trades:,}</div>
                <div class="metric-label">Total Trades</div>
            </div>
        </div>

        <h2>📈 Equity Curves</h2>
        <div class="chart-container">
            {equity_fig}
        </div>

        <h2>🔗 Correlation Matrix</h2>
        <div class="chart-container">
            {corr_fig}
        </div>

        <h2>🎲 Monte Carlo Analysis ({mc_result.num_simulations:,} simulations)</h2>

        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value">${mc_result.final_equity_mean:,.0f}</div>
                <div class="metric-label">Expected Final Equity</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{mc_result.max_dd_mean:.1f}%</div>
                <div class="metric-label">Expected Max DD</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{mc_result.dd_95_ci:.1f}%</div>
                <div class="metric-label">95% DD Confidence</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{mc_result.dd_99_ci:.1f}%</div>
                <div class="metric-label">99% DD Confidence</div>
            </div>
        </div>

        <h3>⚠️ Probability of Ruin</h3>
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value {'risk-low' if mc_result.prob_ruin_10pct < 0.2 else 'risk-medium' if mc_result.prob_ruin_10pct < 0.5 else 'risk-high'}">{mc_result.prob_ruin_10pct*100:.1f}%</div>
                <div class="metric-label">P(DD ≥ 10%)</div>
            </div>
            <div class="metric-card">
                <div class="metric-value {'risk-low' if mc_result.prob_ruin_20pct < 0.1 else 'risk-medium' if mc_result.prob_ruin_20pct < 0.3 else 'risk-high'}">{mc_result.prob_ruin_20pct*100:.1f}%</div>
                <div class="metric-label">P(DD ≥ 20%)</div>
            </div>
            <div class="metric-card">
                <div class="metric-value {'risk-low' if mc_result.prob_ruin_30pct < 0.05 else 'risk-medium' if mc_result.prob_ruin_30pct < 0.15 else 'risk-high'}">{mc_result.prob_ruin_30pct*100:.1f}%</div>
                <div class="metric-label">P(DD ≥ 30%)</div>
            </div>
            <div class="metric-card">
                <div class="metric-value {'risk-low' if mc_result.prob_ruin_50pct < 0.01 else 'risk-medium' if mc_result.prob_ruin_50pct < 0.05 else 'risk-high'}">{mc_result.prob_ruin_50pct*100:.1f}%</div>
                <div class="metric-label">P(DD ≥ 50%)</div>
            </div>
        </div>

        <div class="grid-2">
            <div class="chart-container">
                {mc_fan_fig}
            </div>
            <div class="chart-container">
                {mc_dd_fig}
            </div>
        </div>

        <h2>📊 Strategy Details</h2>
        <table>
            <thead>
                <tr>
                    <th>Strategy</th>
                    <th>Symbol</th>
                    <th>Net Profit</th>
                    <th>Equity DD $</th>
                    <th>Equity DD %</th>
                    <th>Sharpe</th>
                    <th>Profit Factor</th>
                    <th>Return/DD</th>
                    <th>Win Rate</th>
                    <th>Trades</th>
                </tr>
            </thead>
            <tbody>
                {strategy_table}
            </tbody>
        </table>

        <p class="timestamp">Report generated by Portfolio Optimizer</p>
    </div>
</body>
</html>
    """

    return html


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Generate Portfolio Report')
    parser.add_argument('--data-dir', default='data/candidates', help='Directory with MT5 HTML files')
    parser.add_argument('--output', default='portfolio_report.html', help='Output HTML file')
    parser.add_argument('--simulations', type=int, default=1000, help='Number of Monte Carlo simulations')
    parser.add_argument('--balance', type=float, default=10000.0, help='Initial balance')

    args = parser.parse_args()

    generate_report(
        data_dir=args.data_dir,
        output_file=args.output,
        num_monte_carlo=args.simulations,
        initial_balance=args.balance
    )

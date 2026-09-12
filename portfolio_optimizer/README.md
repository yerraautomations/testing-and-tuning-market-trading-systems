# Portfolio Optimizer

A desktop application for optimizing trading strategy portfolios. Parses MT5 backtest reports and finds optimal strategy combinations based on correlation and performance metrics.

## Features

- **MT5 HTML Parser**: Automatically parse Strategy Tester reports from MetaTrader 5
- **Correlation Analysis**: Calculate correlation between strategies using daily returns
- **Portfolio Optimization**: Find optimal strategies to add to your existing portfolio
- **Lock Constraints**: Keep performing strategies locked while finding new additions
- **Visual Dashboard**: Interactive charts, correlation heatmaps, and equity curves

## Project Structure

```
portfolio_optimizer/
├── __init__.py           # Python package init
├── mt5_parser.py         # MT5 HTML report parser
├── correlation.py        # Correlation calculation engine
├── optimizer.py          # Portfolio optimization logic
├── api.py                # API bridge for frontend
├── requirements.txt      # Python dependencies
│
└── desktop-app/          # Tauri + React desktop application
    ├── src/              # React frontend
    │   ├── components/   # UI components
    │   ├── App.tsx       # Main application
    │   └── styles.css    # Styling
    │
    ├── src-tauri/        # Tauri backend (Rust)
    │   ├── src/main.rs   # Tauri commands
    │   └── Cargo.toml    # Rust dependencies
    │
    └── package.json      # Node dependencies
```

## Installation

### Prerequisites

- Python 3.8+
- Node.js 18+
- Rust (for Tauri)

### Python Backend

```bash
cd portfolio_optimizer
pip install -r requirements.txt
```

### Desktop App

```bash
cd portfolio_optimizer/desktop-app
npm install
```

## Usage

### Command Line (Python)

```bash
# Parse a single MT5 HTML file
python mt5_parser.py /path/to/report.html

# Run optimization
python optimizer.py ./locked_strategies ./candidate_strategies
```

### Desktop App

```bash
cd desktop-app
npm run tauri:dev    # Development mode
npm run tauri:build  # Build production .exe
```

## How It Works

1. **Load Locked Strategies**: Upload MT5 HTML reports for strategies you want to keep
2. **Load Candidates**: Upload MT5 HTML reports for strategies you're considering
3. **Configure**: Set your criteria (min Sharpe, max correlation, etc.)
4. **Optimize**: Find the best candidates that complement your locked portfolio
5. **Review**: View correlation heatmap, equity curves, and recommendations

## Optimization Algorithm

The optimizer scores candidates based on:

- **Correlation** (35%): Lower correlation with locked strategies = higher score
- **Sharpe Ratio** (25%): Risk-adjusted returns
- **Profit Factor** (15%): Gross profit / gross loss ratio
- **Drawdown** (15%): Maximum drawdown percentage
- **Recovery Factor** (10%): Net profit / max drawdown

## License

MIT License

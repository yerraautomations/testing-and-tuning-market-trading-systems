# Strategy validation platform (Python)

A Python implementation of the testing methodology from Timothy Masters'
*Testing and Tuning Market Trading Systems* (this repository's C++ code),
extended with a prop-firm evaluation layer. Its job is to answer two
questions honestly, before you risk a challenge fee:

1. **Is this strategy's edge real, or an artefact of optimization?**
2. **If it is real, what is the probability it passes a prop-firm challenge,
   and how should it be sized?**

## Install

```bash
pip install -r requirements.txt          # numpy, pandas (matplotlib optional, for plots)
```

## Quick start

```bash
python data_loader.py                     # fetch + cache real BTC / AAPL data
python validate.py BTC momentum           # the 4-step validation on real data
python mcpt_multi.py                      # selection-bias corrected test across all strategies
python crossmarket.py momentum BTC AAPL   # does the edge generalize across markets?
python propfirm.py                        # FTMO pass probability + leverage sweep
python sizing.py                          # vol-targeting vs FTMO pass rate
python verify_fast.py                     # proves fast == original code
```

## The pipeline

```
data_loader Ã¢â€â‚¬Ã¢â€â‚¬> strategies_library Ã¢â€â‚¬Ã¢â€â‚¬> validate (4 steps) Ã¢â€â‚¬Ã¢â€â‚¬> mcpt_multi Ã¢â€â‚¬Ã¢â€â‚¬> crossmarket
                      Ã¢â€â€š                                                          Ã¢â€â€š
                      Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬> sizing Ã¢â€â‚¬Ã¢â€â‚¬> propfirm (P(pass), sweep) <Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€Ëœ
```

### The 4 validation steps (`validate.py`)

| Step | Question | Module | Pass rule |
|---|---|---|---|
| 1. In-sample excellence | Does the optimized strategy look good, and not *suspiciously* good? | `strategies_library` | judgement |
| 2. In-sample MCPT | Could optimization have found this in noise? | `mcpt_fast.insample_mcpt_parallel` | p < 0.01 |
| 3. Walk-forward | Does it hold up when re-optimized on rolling windows and traded on unseen data? | `mcpt.walkforward_signal` | judgement |
| 4. Walk-forward MCPT | Could a worthless strategy have produced the OOS result by luck? | `mcpt_fast.walkforward_mcpt_parallel` | p < 0.05 (1 yr OOS), p < 0.01 (2+ yrs) |

Permutations shuffle the bars (intra-bar moves and gaps separately) so the
returns keep their distribution but lose their order; a real edge beats the
vast majority of shuffles, a curve-fit one does not.

### Guarding against "I tried many things" (`mcpt_multi.py`)

Testing N strategies and keeping the one that passes reintroduces selection
bias. `multi_system_mcpt` optimizes *every* candidate on *every* permutation
and compares each real result to the distribution of the **best of N** on
noise. A strategy that "passes alone but fails after correction" was the
lucky one.

### Cross-market check (`crossmarket.py`)

Runs one strategy on many markets. Passing on exactly one of many related
markets is the fingerprint of luck; a real edge tends to show up at least
directionally on related instruments.

### Prop-firm layer (`propfirm.py`, `sizing.py`)

`propfirm.py` encodes the rules (FTMO 2-step / 1-step presets, or your own
`PropFirmRules`): profit targets per phase, max daily loss (measured per
calendar day in the firm's timezone), max loss (static or trailing), minimum
trading days, optional time limit. It applies leverage and transaction costs
(`CostModel`, with a stress multiplier), then reports

* **historical pass rate** -- start a challenge on every historical day;
* **bootstrap pass rate** -- block-bootstrap Monte Carlo over a horizon,
  e.g. "P(funded within 60 days)";
* failure breakdown: daily-loss breach vs max-loss breach vs time;
* a **leverage sweep** showing the trade-off between P(pass) and days to fund.

`sizing.py` adds volatility targeting: it scales the position so the
account's expected volatility stays near a target derived from the daily-loss
rule (e.g. "a 3-sigma day stays inside 5%"). With no time limit on FTMO
challenges, trading smaller and taking longer is usually the biggest single
improvement in pass probability.

## Adding your own strategies

Copy `user_strategies_template.py` to `user_strategies.py`, implement
`signal(close, params)` and `optimize(close, objective_func)` for each
strategy, and call `register_strategy(...)`. The scripts pick the file up
automatically. Rules: no look-ahead, module-level functions, vectorized.

## Data

* `data_loader.load("BTC" | "AAPL")` -- cached real daily data (GitHub-hosted
  sources; exchange APIs are often firewalled).
* `data_loader.load_mt5(path, tz="UTC")` -- MT5 / QuantDataManager bar
  exports (M1 etc.); `resample_ohlc()` and `resample_daily_session()` build
  higher timeframes, including 17:00-New-York daily bars.
* `data_loader.load_csv(path)` -- any CSV with date/open/high/low/close.

Store everything in **UTC**; apply session and daily-reset boundaries in code.

## Discipline (the part software cannot do for you)

* Split data into development and a **lockbox** you touch once, at the end.
* Prefer strategies with an economic rationale; it shrinks the search space
  and the multiple-testing penalty.
* Costs matter more than entries for FX/prop challenges. Stress them.
* Passing every test is evidence of a real edge, not a guarantee of future
  profit. It mainly stops you paying to trade noise.

## Files

| File | Purpose |
|---|---|
| `permutation.py` | Bar permutation (single and multi-market) |
| `objectives.py` | Profit factor, Sharpe, per-bar strategy returns |
| `mcpt.py` | Serial in-sample / walk-forward MCPT, walk-forward signal |
| `mcpt_fast.py` | Parallel MCPT (identical results, all cores) |
| `mcpt_multi.py` | Multiple-system (selection-bias corrected) MCPT |
| `strategies.py`, `strategies_fast.py` | Original and vectorized demo strategies |
| `strategies_library.py` | Strategy registry + 5 vectorized strategies + adapters |
| `user_strategies_template.py` | Template for your own strategies |
| `validate.py` | 4-step validation runner |
| `crossmarket.py` | Cross-market robustness report |
| `propfirm.py` | Prop-firm challenge simulator, P(pass), leverage sweep |
| `sizing.py` | Volatility targeting |
| `report.py` | JSON export and PNG plots |
| `data_loader.py` | Data fetching, caching, MT5/QDM loading, resampling |
| `verify_fast.py` | Equivalence proof for the fast path |
| `example_donchian.py` | Original synthetic-data walkthrough |
| `to_optimizer.py` | Bridge: platform signals -> optimizer strategies / databank (see `portfolio_optimizer/headless.py`) |
| `instruments.py` | Instrument profiles (forex / futures / crypto): costs, contract math, bars per year (`config/instruments.json`) |
| `bank.py` | Strategy bank: folders of passing strategies, criteria, run log, databank export |
| `gate.py` | The full gate for one strategy (IS MCPT, walk-forward, WF MCPT, lockbox, cross-market, prop-firm sim) and Tier B for MT5 reports |
| `pipeline.py` | Config-driven run: all strategies through the gate, multiple-system correction, criteria, bank, optimizer |
| `parity.py` | Trade-by-trade comparison of an MT5 EA report with its Python port |

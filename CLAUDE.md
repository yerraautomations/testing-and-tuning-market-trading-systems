# Project brief for Claude (and humans)

This file is the hand-off of a long design conversation. Read it first.

## What this repository is

- Root: Timothy Masters' original C++ code for *Testing and Tuning Market
  Trading Systems* (Apress, 2018). Reference only; we don't build it.
- `python/`: our platform -- a Python implementation of the book's
  permutation-based validation methods, extended with a prop-firm evaluation
  layer. `python/README.md` documents modules and commands.
- `portfolio_optimizer/`: the user's Streamlit portfolio optimizer (restored
  from a lost branch), plus `headless.py` so validated strategies flow into
  it without the UI. See "Portfolio optimizer" below.

Branch for all work: `claude/explore-trading-systems-FbLc0`.

## Where the work happens now (local Windows)

- The repo is cloned at `C:\Users\abeny\testing-and-tuning-market-trading-systems`
  and Claude Code runs locally from that folder (Python 3.12, pandas 3.x,
  Windows 10, PowerShell). The earlier cloud sandbox is retired.
- The machine has 88 logical cores in two processor groups. **Windows caps a
  ProcessPoolExecutor at 61 workers**; every pool in this repo is capped
  (`min(os.cpu_count(), 61)`). Keep doing that in new code.
- Raw exports live in `python/data/raw/` (git-ignored). Tick-derived bar
  caches are `python/data/*_M1.csv` (git-ignored, rebuilt by `load_ticks`).
- Big files: never `Get-Content` a multi-GB CSV to count lines; stream it in
  Python instead.
- git identity is set repo-locally; `git push` needs the user's interactive
  GitHub login (run it in their own PowerShell window).

## The user's goal

Pass prop-firm challenges (FTMO first; futures or forex, either is fine),
consistently, with strategies they build themselves. They have written EAs in
MT5 (34 of them in `Downloads\Final_EAs.zip`: Market Regime KS / Wasserstein /
CSSA / CUSUM / DTW / Adaptive families, Breakout, Pivot Bounce, Fibonacci
Scalp, News Retracement, Volume Profile, DOM, Banker Flow) and found MT5
backtests "not enough" -- the MT5 optimizer is a curve-fitting machine with
no honesty check. This platform is the honesty check. The end state they
want: **strategies validated by the platform are sent to the portfolio
optimizer automatically and portfolios come out, with no manual steps.**

Agreed plan ("what I would do in your shoes"): build a diversified baseline
of boring edges (trend / breakout on H1-H4 across the 7 majors and the micro
futures) sized to the daily-loss rule first; then triage the user's EAs
(port only the price-based, distinct ones) against that baseline; lockbox
once; shadow-trade on an FTMO demo before paying a fee. Instant-funding vs
challenge is decided by expected payout per fee from the simulator, not by
"ease" (an instant-funding preset is a planned addition).

## Principles agreed in the conversation (do not relitigate)

1. **Permutation tests are filters, not generators.** They reject curve-fit
   strategies; they cannot manufacture an edge. Passing is evidence of a real
   edge, never a guarantee of profit.
2. **Searching many strategies reintroduces selection bias.** Use
   `mcpt_multi.py` (best-of-N null) whenever more than one candidate was
   tried. Keep a **lockbox** holdout (last ~3 years) touched exactly once.
3. **For prop firms the binding constraint is the daily-loss limit**, not the
   entry signal. Position sizing (`sizing.py`) and realistic spreads matter
   more than the strategy logic. With no FTMO time limit, "trade smaller,
   take longer" is the biggest lever.
4. **A strategy that passes on one market and fails on related ones is
   guilty until proven innocent** unless the mechanism was named *before*
   looking (`crossmarket.py`).
5. **Costs must be stressed** (1x / 2x / 3x spread) before trusting an FX
   result.
6. **Never overpromise.** Report P(pass) as a probability with its failure
   breakdown; markets have bad periods.

## Data conventions

- Store all bar data in **UTC**. Apply session boundaries and the prop-firm
  daily reset (FTMO: CE(S)T calendar day) in code, never baked into files.
- **FTMO tick export** (`python/data/raw/2026.9.2EURUSD_FTMO-TICK-EURUSD_ftmo.csv`,
  12 GB, 278.6M ticks, 2016-07-04 .. 2026-08-28): columns
  `DateTime,Bid,Ask,Volume`, timestamps `YYYYMMDD HH:MM:SS.fff` in **broker
  time** (GMT+2/+3 switching on US DST = New York + 7h; verified: every
  weekend gap is Fri 23:59 -> Mon 00:00). `data_loader.load_ticks(path)`
  streams it into 3.78M **M1 bid bars with per-bar spread** (mean/min/max/
  close), converts to UTC and caches to `python/data/<stem>_M1.csv`.
  Coverage is complete (~373k bars per full year; only holiday gaps).
- Also in `python/data/raw/` (copied from Downloads, all load with
  `load_mt5`): `EURUSD_M1_202301020212_202505131901.csv` (another broker,
  2023-2025, broker time, median spread 1.7 pips -- much wider than FTMO) and
  seven CME micro-futures 1-minute continuous files 2024-01 .. 2025-06
  (`MES, MNQ, M2K, MYM, M6E, M6A, M6B_continuous.csv`, timestamps UTC).
  `Downloads\EURUSD_202301020212_202505131901.csv` (2.4 GB) is an MT5 tick
  export (`<DATE> <TIME> <BID> <ASK> <LAST> <VOLUME> <FLAGS>`), loadable with
  a small adapter to `load_ticks` if ever needed.
- QuantDataManager Dukascopy M1 for the 7 majors (2003+) was downloaded into
  QDM but **never exported to CSV** -- export from QDM (All time, M1, fixed
  ~1 pip spread, target timezone UTC) into `python/data/raw/` when needed.
- Built-in cached datasets: `python/data/BTC.csv` (daily close-only,
  2010-2026; Open=High=Low=Close by design) and `AAPL.csv` (daily OHLC).

## FTMO EUR/USD spread facts (measured from the ticks, in pips)

- Tick-weighted median **0.30**, p75 0.39, p90 0.50, p99 1.35; mean 0.35.
  A fixed 1-pip assumption is ~3x too pessimistic for EUR/USD at FTMO;
  use the per-bar `Spread` column as the 1x baseline and stress 2x / 3x.
- By session (UTC): London 07-11 and overlap 12-16 ~0.29 mean; Asia 0.35;
  NY afternoon 0.33; **rollover 21:00-23:59 mean ~1.0, hour 21 mean 1.5 /
  p90 3.4**. Avoid entering or holding pending orders through 21:00-23:00
  UTC. Holiday sessions (Christmas, New Year) show 20-34 pip spreads.
- 3.5% of minutes average > 1 pip; 0.45% have a max > 5 pips.

## State of the platform (`python/`, all tested on real data)

| Layer | Module | Status |
|---|---|---|
| Bar permutation, objectives | `permutation.py`, `objectives.py` | done |
| 4-step validation (IS excellence, IS MCPT, walk-forward, WF MCPT) | `mcpt.py`, `validate.py` | done |
| Fast path (vectorized + parallel, proven identical) | `strategies_fast.py`, `mcpt_fast.py`, `verify_fast.py` | done |
| Strategy registry + 5 demo strategies + plug-in template | `strategies_library.py`, `user_strategies_template.py` | done |
| Multiple-system (selection-bias corrected) MCPT | `mcpt_multi.py` | done |
| Cross-market report | `crossmarket.py` | done |
| Prop-firm simulator: FTMO rules, costs, P(pass), leverage sweep | `propfirm.py` | done |
| Volatility targeting | `sizing.py` | done |
| JSON export + PNG plots | `report.py` | done |
| MT5/QDM bar loader, resampling (incl. 17:00 NY sessions) | `data_loader.py` | done, verified on real MT5 + futures files |
| Tick loader (bid/ask -> M1 bars + spread) + `spread_report` | `data_loader.load_ticks` | done, run on the real 12 GB FTMO file |
| Bridge to the optimizer (signal -> MT5-style deals -> databank) | `to_optimizer.py` | done, demo runs end to end |
| Per-bar spread in the cost model (`propfirm.CostModel` takes one constant) | -- | **next** |
| Baseline diversified portfolio (majors + micro futures) through the gate + FTMO sim | -- | next |
| Port the user's price-based EAs (Breakout, Pivot Bounce, one Market Regime, Fib Scalp) | `user_strategies.py` | next |
| Instant-funding preset (trailing DD, payout threshold) + EV-per-fee comparison | `propfirm.py` | planned |
| Drawdown-aware cut-off in sizing (vol-targeting shifts failures to max-loss) | -- | planned |

Key demo results (BTC, momentum, walk-forward OOS): at 1x leverage FTMO
P(pass) = 0.2% (90% breach the daily limit); vol-targeting at 24% -> ~36%
with median 69 days. Multi-system test: donchian passes alone (p=0.005) but
fails after correction (p=0.025); momentum and ma_cross survive.

## Portfolio optimizer (`portfolio_optimizer/`)

Streamlit app that loads MT5 Strategy Tester HTML reports (or JSON
databanks), computes correlations from daily returns, searches strategy
combinations under filters/constraints, ranks them (Return/DD etc.), runs
Monte Carlo, and exports Excel/HTML reports. Restored 2026-09-11 from
`Downloads\Portfolio Optimizer` (the GitHub branch had been deleted); the
build/ and dist/ folders are git-ignored (rebuild with `build_exe.bat`).

Fixes verified present or re-applied (see commit "Restore portfolio
optimizer"): trade.time, scipy/xlsxwriter in requirements, Excel attribute
names, Stability/Symmetry from closed deals only, portfolio selector before
the export buttons, HTML report mirroring the Analysis tab (negative
drawdown, monthly bar chart), batch launcher, EXE launcher/spec fixes.
New: 61-worker pool cap everywhere; **same-named reports are made unique**
(`mt5_parser.make_names_unique`, suffix = report file stem) because the
correlation engine keys by name and silently merged them.

Run: `cd portfolio_optimizer` then `streamlit run streamlit_app.py`, or
double-click `Launch_Portfolio_Optimizer.bat`. Dependencies:
`pip install -r portfolio_optimizer/requirements.txt`.

**Headless pipeline (the automation the user wants):**

```bash
# validated platform strategies -> databank (see python/to_optimizer.py demo)
python python/to_optimizer.py
# any mix of MT5 report folders and databanks -> ranked portfolios + reports
python portfolio_optimizer/headless.py --databank platform_btc_demo --html data/candidates \
    --min-strategies 2 --max-strategies 4 --keep-top 20 --name run1 --export-top 3
```
Outputs: `saved_portfolios/<name>.json` (loadable in the UI's Results tab)
and `reports/<name>/` (JSON summary, HTML + Excel per top portfolio).
"No portfolios found" is usually the `--min-trades` filter (default 30).

## How to run (platform)

```bash
cd python && pip install -r requirements.txt   # + matplotlib for plots
python validate.py BTC momentum
python mcpt_multi.py
python crossmarket.py momentum BTC AAPL
python propfirm.py
python sizing.py
python data_loader.py ticks data/raw/<tick file>.csv   # build/reuse M1 cache + spread report (~8 min for 12 GB)
python to_optimizer.py                                # demo bridge into the optimizer
```

A `checkpoint-before-speedup` git tag marks the state before the fast path
was added.

## Working style the user prefers

Concise, honest, decisive recommendations; keep originals intact and changes
reversible; test in the terminal and report real outputs; ask for the exact
data format before writing a loader. The user is new to terminals: give
step-by-step commands and say what the output should look like. Do not
search the whole PC for files -- ask, or look in Downloads.
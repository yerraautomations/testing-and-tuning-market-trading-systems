# Project brief for Claude (and humans)

This file is the hand-off of a long design conversation. Read it first.

## What this repository is

- Root: Timothy Masters' original C++ code for *Testing and Tuning Market
  Trading Systems* (Apress, 2018). Reference only; we don't build it.
- `python/`: our platform -- a Python implementation of the book's
  permutation-based validation methods, extended with a prop-firm evaluation
  layer. `python/README.md` documents modules and commands.

Branch for all work: `claude/explore-trading-systems-FbLc0`.

## Where the work happens now (local Windows)

- The repo is cloned at `C:\Users\abeny\testing-and-tuning-market-trading-systems`
  and Claude Code runs locally from that folder (Python 3.12, pandas 3.x,
  Windows 10, PowerShell). The earlier cloud sandbox is retired.
- Raw exports live in `python/data/raw/` (git-ignored). Tick-derived bar
  caches are `python/data/*_M1.csv` (git-ignored, rebuilt by `load_ticks`).
- Big files: never `Get-Content` a multi-GB CSV to count lines; stream it in
  Python instead.

## The user's goal

Pass prop-firm challenges (FTMO first; futures or forex, either is fine),
consistently, with strategies they build themselves. They have written EAs in
MT5 and found MT5 backtests "not enough" -- the MT5 optimizer is a
curve-fitting machine with no honesty check. This platform is the honesty
check. They are consolidating their own strategies in another chat and will
plug them in via `python/user_strategies.py` (see
`user_strategies_template.py`).

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
- QuantDataManager Dukascopy M1 exports for the 7 majors (2003+) are the
  research history. Agreed export settings: All time, M1, fixed ~1 pip
  spread, target timezone UTC. `data_loader.load_mt5()` reads them.
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

## State of the platform (all tested on real data)

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
| MT5/QDM bar loader, resampling (incl. 17:00 NY sessions) | `data_loader.py` | done, tested on synthetic MT5 files only |
| Tick loader (bid/ask -> M1 bars + spread) + `spread_report` | `data_loader.load_ticks` | **done, run on the real 12 GB FTMO file** |
| Per-bar spread in the cost model (`propfirm.CostModel` takes one constant) | -- | **next** |
| Load the Dukascopy M1 majors, then plug in the user's strategies | -- | next |
| Drawdown-aware cut-off in sizing (vol-targeting shifts failures to max-loss) | -- | planned |

Key demo results (BTC, momentum, walk-forward OOS): at 1x leverage FTMO
P(pass) = 0.2% (90% breach the daily limit); vol-targeting at 24% -> ~36%
with median 69 days. Multi-system test: donchian passes alone (p=0.005) but
fails after correction (p=0.025); momentum and ma_cross survive.

## How to run

```bash
cd python && pip install -r requirements.txt   # + matplotlib for plots
python validate.py BTC momentum
python mcpt_multi.py
python crossmarket.py momentum BTC AAPL
python propfirm.py
python sizing.py
python data_loader.py ticks data/raw/<tick file>.csv   # build/reuse M1 cache + spread report (~8 min for 12 GB)
```

A `checkpoint-before-speedup` git tag marks the state before the fast path
was added.

## Working style the user prefers

Concise, honest, decisive recommendations; keep originals intact and changes
reversible; test in the terminal and report real outputs; ask for the exact
data format before writing a loader. The user is new to terminals: give
step-by-step commands and say what the output should look like.
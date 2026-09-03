# Project brief for Claude (and humans)

This file is the hand-off of a long design conversation. Read it first.

## What this repository is

- Root: Timothy Masters' original C++ code for *Testing and Tuning Market
  Trading Systems* (Apress, 2018). Reference only; we don't build it.
- `python/`: our platform -- a Python implementation of the book's
  permutation-based validation methods, extended with a prop-firm evaluation
  layer. `python/README.md` documents modules and commands.

Branch for all work: `claude/explore-trading-systems-FbLc0`.

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
- The user downloads data with **QuantDataManager** (Dukascopy M1 for the 7
  majors back to 2003; broker profiles use EETUS = GMT+2/+3 with US DST).
  Agreed export settings: All time, M1, fixed ~1 pip spread, target
  timezone UTC. `data_loader.load_mt5()` reads MT5-style bar exports.
- The user also has an **FTMO tick export** (`...EURUSD_FTMO-TICK-...`).
  A tick loader (time, bid, ask -> M1 bid OHLC + per-bar spread) is the next
  thing to build; ticks give FTMO's *actual* spread.
- Built-in cached datasets: `python/data/BTC.csv` (daily close-only,
  2010-2026; Open=High=Low=Close by design) and `AAPL.csv` (daily OHLC).

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
| Tick loader (bid/ask -> bars + spread) | -- | **next** |
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
```

Environment notes: the cloud sandbox firewalls exchange APIs but allows
GitHub; locally none of that applies. A `checkpoint-before-speedup` git tag
marks the state before the fast path was added.

## Working style the user prefers

Concise, honest, decisive recommendations; keep originals intact and changes
reversible; test in the terminal and report real outputs; ask for the exact
data format before writing a loader.

# Project brief for Claude (and humans)

This file is the hand-off of a long design conversation. Read it first.

## What this repository is

- Root: Timothy Masters' original C++ code for *Testing and Tuning Market
  Trading Systems* (Apress, 2018). Reference only; we don't build it.
- `python/`: the validation platform -- permutation-based honesty tests,
  walk-forward, prop-firm simulation, the gate/pipeline, the strategy bank.
  `python/README.md` lists every module.
- `portfolio_optimizer/`: the user's Streamlit portfolio optimizer (restored
  from a lost branch, never modified beyond bug fixes) plus `headless.py`.
- `app/`: the two-page **Trading Workbench** (Strategy Tester + Portfolio
  Builder). Launch with `Launch_App.bat` or `streamlit run app/Home.py`.
- `config/`: `propfirms.json` (rule presets), `instruments.json` (profiles),
  `pipeline_example.json` (a pipeline config to copy).
- `strategy_bank/` (git-ignored, local data): folders of strategies that
  PASSED the gate, plus `_runs.csv` (every attempt) and `_runs/*.json`.

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
- Never import `portfolio_optimizer/streamlit_app.py` from inside a Streamlit
  page (its module-level `st.set_page_config` would fire mid-page). The
  workbench runs it via `runpy` as its own page; bank/pipeline code uses
  `bank.py` for databank I/O instead of importing the app.

## The user's goal

Pass prop-firm challenges (FTMO first; futures or forex, either is fine),
consistently, with strategies they build themselves. They have written EAs in
MT5 (34 of them in `Downloads\Final_EAs.zip`: Market Regime KS / Wasserstein /
CSSA / CUSUM / DTW / Adaptive families, Breakout, Pivot Bounce, Fibonacci
Scalp, News Retracement, Volume Profile, DOM, Banker Flow) and found MT5
backtests "not enough" -- the MT5 optimizer is a curve-fitting machine with
no honesty check. This platform is the honesty check. The end state:
**strategies validated by the platform flow into the portfolio optimizer
automatically; the user only chooses criteria, folders and what to send.**

Agreed plan: build a diversified baseline of boring edges (trend / breakout
on H1-H4 across the 7 majors and the micro futures) sized to the daily-loss
rule first; then triage the user's EAs (port only the price-based, distinct
ones) against that baseline; lockbox once; shadow-trade on a demo before
paying a fee. Instant-funding vs challenge is decided by expected payout per
fee from the simulator, not by "ease".

## Principles agreed in the conversation (do not relitigate)

1. **Permutation tests are filters, not generators.** They reject curve-fit
   strategies; they cannot manufacture an edge. Passing is evidence of a real
   edge, never a guarantee of profit.
2. **Searching many strategies reintroduces selection bias.** The pipeline
   always runs the multiple-system test over every candidate of a run and
   checks the corrected p. Every attempt is logged (`strategy_bank/_runs.csv`)
   even though only passers are kept. Keep a **lockbox** (last ~2-3 years)
   scored exactly once -- the pipeline warns when it has been scored before.
3. **For prop firms the binding constraint is the daily-loss limit**, not the
   entry signal. Position sizing (`sizing.py`) and realistic spreads matter
   more than the strategy logic. "Trade smaller, take longer" is the biggest
   lever.
4. **A strategy that passes on one market and fails on related ones is
   guilty until proven innocent** unless the mechanism was named *before*
   looking (`crossmarket.py`, or `extra_markets` in the pipeline).
5. **Costs must be stressed** (1x / 2x / 3x spread) before trusting an FX
   result (`gate.cost_stress`).
6. **Never overpromise.** Report P(pass) as a probability with its failure
   breakdown; markets have bad periods.
7. **The MQL5 EA is what trades; the Python port is what gets validated.**
   Prove the port with `parity.py` (trade-by-trade vs the EA's MT5 report on
   the same bars) before trusting the validation, then shadow-trade on a demo.

## Strategy formats and tiers

- **Tier A** = a Python strategy (`signal(close, params)` + `optimize`) in
  `strategies_library.py` or `user_strategies.py`. Gets the full gate. The
  tester picks the parameters (grid declared in the strategy); the user does
  not hand-pick them -- that is the point.
- **Tier B** = an MT5 Strategy Tester HTML report. Only the prop-firm
  simulation and Monte Carlo on its trade list; saved as an *unvalidated*
  strategy (its parameters were chosen by MT5's optimizer).
- Both are stored in the optimizer's databank JSON format (`bank.py`), so
  any bank selection becomes a databank for the Portfolio Builder or
  `headless.py`. MT5 HTML remains an accepted input of the builder.

## Data conventions

- Store all bar data in **UTC**. Apply session boundaries and the prop-firm
  daily reset in code, never baked into files.
- **FTMO tick export** (`python/data/raw/2026.9.2EURUSD_FTMO-TICK-EURUSD_ftmo.csv`,
  12 GB, 278.6M ticks, 2016-07-04 .. 2026-08-28): `DateTime,Bid,Ask,Volume`,
  timestamps in **broker time** (GMT+2/+3 on US DST = New York + 7h;
  verified by the weekend gaps). `data_loader.load_ticks(path)` -> 3.78M M1
  bid bars with per-bar spread, cached as `python/data/<stem>_M1.csv`. The
  gate uses that Spread column bar by bar when present.
- Also in `python/data/raw/` (all load with `load_mt5`): another broker's
  `EURUSD_M1_202301020212_202505131901.csv` (2023-2025, broker time, median
  spread 1.7 pips) and seven CME micro-futures 1-minute continuous files
  2024-01 .. 2025-06 (`MES, MNQ, M2K, MYM, M6E, M6A, M6B_continuous.csv`,
  UTC). `Downloads\EURUSD_202301020212_202505131901.csv` (2.4 GB) is an MT5
  tick export, loadable with a small adapter if ever needed.
- QuantDataManager Dukascopy M1 for the 7 majors (2003+) is downloaded into
  QDM but **not exported to CSV yet** (All time, M1, target timezone UTC).
- For MT5 <-> Python parity use bars exported from MT5 itself (Symbols ->
  Bars -> Export) so both sides see identical bars; QDM data can also be
  imported into MT5 as a custom symbol (export with timezone = Original).
- Built-in cached datasets: `python/data/BTC.csv` (daily close-only) and
  `AAPL.csv` (daily OHLC).

## FTMO EUR/USD spread facts (measured from the ticks, in pips)

- Tick-weighted median **0.30**, p75 0.39, p90 0.50, p99 1.35; mean 0.35.
  A fixed 1-pip assumption is ~3x too pessimistic for EUR/USD at FTMO.
- London 07-11 and overlap 12-16 UTC ~0.29 mean; Asia 0.35; **rollover
  21:00-23:59 mean ~1.0, hour 21 mean 1.5 / p90 3.4**. Avoid entering or
  holding pending orders through 21:00-23:00 UTC. Holidays: 20-34 pips.

## State of the platform (all tested on real data)

| Layer | Module | Status |
|---|---|---|
| Bar permutation, objectives | `permutation.py`, `objectives.py` | done |
| 4-step validation, fast parallel path, multiple-system MCPT, cross-market | `mcpt*.py`, `validate.py`, `crossmarket.py` | done |
| Strategy registry + demo strategies + user template | `strategies_library.py`, `user_strategies_template.py` | done |
| Prop-firm simulator, **generic rules + presets** (FTMO verified; others approximate) | `propfirm.py`, `config/propfirms.json` | done |
| Instrument profiles (forex / futures / crypto) | `instruments.py`, `config/instruments.json` | done |
| Volatility targeting | `sizing.py` | done |
| Data loaders: cached, MT5 bars, ticks, resampling | `data_loader.py` | done |
| Bridge to the optimizer | `to_optimizer.py` | done |
| Strategy bank (folders, criteria, run log, databank export) | `bank.py` | done |
| The gate (Tier A) and Tier B | `gate.py` | done |
| Pipeline (config-driven, correction + lockbox built in, optimizer hand-off) | `pipeline.py`, `config/pipeline_example.json` | done |
| Parity tool (EA report vs Python port) | `parity.py` | done, alignment self-tested; awaits a real EA report |
| Workbench app (Strategy Tester + Portfolio Builder pages) | `app/` | done, pages render; not yet clicked through by the user |
| Portfolio optimizer + headless pipeline | `portfolio_optimizer/` | done |
| Baseline diversified portfolio (majors + micro futures) through the gate | -- | **next** |
| Port the user's price-based EAs, prove with parity, run the gate | `user_strategies.py` | next |
| Drawdown-aware cut-off in sizing (vol-targeting shifts failures to max-loss) | -- | planned |

Honest demo results: with the default criteria (corrected p <= 0.01, WF p
<= 0.05, lockbox PF >= 1, P(pass) >= 0.2) **none of the five demo
strategies on daily BTC pass** -- momentum is closest (corrected p 0.023,
lockbox PF 1.02). That is the gate working, not a bug. A relaxed run
(corrected p <= 0.05, lockbox PF >= 0.9) is kept in bank folder `demo`.

## How to run

```bash
Launch_App.bat                                  # the workbench (both pages)
cd python
python pipeline.py --config ../config/pipeline_example.json   # everything, no UI
python parity.py --report R.html --bars bars.csv --strategy my_breakout --params "[40,0.002]"
python data_loader.py ticks data/raw/<tick file>.csv          # rebuild M1 cache + spread report
python validate.py BTC momentum | mcpt_multi.py | crossmarket.py ... | propfirm.py | sizing.py
cd ../portfolio_optimizer
python headless.py --databank <name> --html data/candidates --min-strategies 2 --max-strategies 4
streamlit run streamlit_app.py                  # the optimizer alone
```

`checkpoint-before-speedup` git tag = state before the fast path was added.

## Working style the user prefers

Concise, honest, decisive recommendations; keep originals intact and changes
reversible; test in the terminal and report real outputs; ask for the exact
data format before writing a loader. The user is new to terminals: give
step-by-step commands and say what the output should look like. Do not
search the whole PC for files -- ask, or look in Downloads.
"""
Headless portfolio optimizer: strategies in, ranked portfolios and reports out.

Runs the same portfolio search as the Streamlit "Portfolio Builder" tab
without the UI, so a pipeline can call it after strategies are validated.

    python headless.py --html data/candidates --databank platform_btc_demo \
        --min-strategies 2 --max-strategies 4 --keep-top 20 \
        --name auto_20260911 --out reports/auto_20260911

Inputs
    --html DIR        folder of MT5 Strategy Tester HTML reports (repeatable)
    --databank NAME   a databank saved by the app or by python/to_optimizer.py
                      (repeatable; its locked strategies stay locked)
    --lock NAME       lock a strategy by (unique) name (repeatable)

Outputs (in --out, plus saved_portfolios/<name>.json for the UI)
    portfolios_<name>.json   ranked portfolios + settings + run info
    portfolio_<k>.html/.xlsx the Analysis-tab report for the top --export-top

Notes
    * Filters default to the app's defaults; "Min Total Trades" (30) is the
      usual reason for "no portfolios found" -- lower it with --min-trades.
    * Search is exhaustive when the number of combinations is at most
      --max-iterations, otherwise random sampling of that many combinations.
    * Evaluation runs in a process pool (capped at 61 workers on Windows).
"""

import argparse
import itertools
import json
import logging
import math
import os
import random
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)          # silence Streamlit "bare mode" noise
import streamlit_app as app               # noqa: E402  UI only runs under __main__
from mt5_parser import MT5Parser, make_names_unique   # noqa: E402
from correlation import CorrelationEngine             # noqa: E402


RANKERS = {
    "Return/DD": (lambda p: p.return_dd_ratio, True),
    "Sharpe": (lambda p: p.sharpe, True),
    "Net Profit": (lambda p: p.total_profit, True),
    "Lowest DD": (lambda p: p.equity_dd, False),
    "Lowest Correlation": (lambda p: p.avg_correlation, False),
}

DEFAULT_FILTERS = dict(min_trades=30, min_pf=1.0, min_sharpe=0.0,
                       max_dd_pct=50.0, min_profit=None)
DEFAULT_SETTINGS = dict(min_strategies=2, max_strategies=5, max_per_symbol=2,
                        max_correlation=0.7, max_equity_dd=0.0,
                        max_balance_dd=0.0, rank_by="Return/DD", keep_top=50,
                        max_iterations=50000, seed=0)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_strategies(html_dirs=(), databanks=(), lock_names=()):
    """Return (strategies, locked_indices) from HTML folders and databanks."""
    strategies, locked = [], set()
    for d in html_dirs:
        found = MT5Parser().parse_directory(str(d), parallel=True)
        print(f"  {len(found)} reports from {d}")
        strategies += found
    for name in databanks:
        found, l = app.load_databank(name)
        offset = len(strategies)
        locked |= {offset + i for i in l}
        print(f"  {len(found)} strategies from databank '{name}' ({len(l)} locked)")
        strategies += found
    make_names_unique(strategies)
    for name in lock_names:
        hits = [i for i, s in enumerate(strategies) if s.name == name]
        if not hits:
            raise SystemExit(f"--lock: no strategy named {name!r}")
        locked |= set(hits)
    for i in locked:
        strategies[i].is_locked = True
    return strategies, locked


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def passes_filters(s, f):
    m = s.metrics
    return (m.total_trades >= f["min_trades"]
            and m.profit_factor >= f["min_pf"]
            and m.sharpe_ratio >= f["min_sharpe"]
            and m.equity_dd_maximal_pct <= f["max_dd_pct"]
            and (f["min_profit"] is None or m.total_net_profit >= f["min_profit"]))


def filter_report(strategies, f):
    """Count how many strategies each filter rejects (diagnostics)."""
    ms = [s.metrics for s in strategies]
    return {
        "total": len(ms),
        f"trades < {f['min_trades']}": sum(m.total_trades < f["min_trades"] for m in ms),
        f"profit factor < {f['min_pf']}": sum(m.profit_factor < f["min_pf"] for m in ms),
        f"sharpe < {f['min_sharpe']}": sum(m.sharpe_ratio < f["min_sharpe"] for m in ms),
        f"equity DD% > {f['max_dd_pct']}": sum(m.equity_dd_maximal_pct > f["max_dd_pct"] for m in ms),
    }


def _symbols_ok(indices, strategies, max_per_symbol):
    count = {}
    for i in indices:
        sym = strategies[i].symbol
        count[sym] = count.get(sym, 0) + 1
        if count[sym] > max_per_symbol:
            return False
    return True


_STATE = {}


def _init(strategies, engine):
    _STATE["s"] = strategies
    _STATE["e"] = engine


def _evaluate(indices):
    return app.calculate_portfolio_metrics(list(indices), _STATE["s"], _STATE["e"])


def generate_portfolios(strategies, locked, filters=None, settings=None,
                        workers=None, verbose=True):
    """Port of the app's generate_portfolios() without any UI dependency."""
    f = {**DEFAULT_FILTERS, **(filters or {})}
    st = {**DEFAULT_SETTINGS, **(settings or {})}
    t0 = time.time()

    engine = CorrelationEngine(strategies)
    engine.build_correlation_matrix()

    candidates = [i for i in range(len(strategies))
                  if i not in locked and passes_filters(strategies[i], f)]
    locked_list = sorted(locked)
    min_add = max(0, st["min_strategies"] - len(locked_list))
    max_add = st["max_strategies"] - len(locked_list)
    info = {"candidates": len(candidates), "locked": len(locked_list),
            "filter_report": filter_report(strategies, f)}
    if verbose:
        print(f"  {len(candidates)} candidates pass filters, {len(locked_list)} locked")
    if max_add < 0 or not candidates and min_add > 0:
        info["mode"] = "none"
        return [], info

    sizes = [n for n in range(min_add, max_add + 1) if n <= len(candidates)]
    total = sum(math.comb(len(candidates), n) for n in sizes)
    if total <= st["max_iterations"]:
        combos = [tuple(locked_list) + c for n in sizes
                  for c in itertools.combinations(candidates, n)]
        info["mode"] = "exhaustive"
    else:
        rng = random.Random(st["seed"])
        seen, attempts = set(), 0
        while len(seen) < st["max_iterations"] and attempts < st["max_iterations"] * 5:
            n = rng.choice(sizes)
            seen.add(tuple(sorted(rng.sample(candidates, n))))
            attempts += 1
        combos = [tuple(locked_list) + c for c in seen]
        info["mode"] = "sampled"
    combos = [c for c in combos if _symbols_ok(c, strategies, st["max_per_symbol"])]
    info["combinations_total"] = total
    info["combinations_evaluated"] = len(combos)
    if verbose:
        print(f"  {info['mode']} search: evaluating {len(combos):,} of "
              f"{total:,} combinations")

    if workers is None:
        workers = min(os.cpu_count() or 1, 61)
    if workers > 1 and len(combos) > 200:
        chunk = max(1, len(combos) // (workers * 8))
        with ProcessPoolExecutor(max_workers=workers, initializer=_init,
                                 initargs=(strategies, engine)) as ex:
            evaluated = list(ex.map(_evaluate, combos, chunksize=chunk))
    else:
        _init(strategies, engine)
        evaluated = [_evaluate(c) for c in combos]

    kept = []
    for p in evaluated:
        if st["max_correlation"] < 1.0 and p.avg_correlation > st["max_correlation"]:
            continue
        if st["max_equity_dd"] > 0 and p.equity_dd > st["max_equity_dd"]:
            continue
        if st["max_balance_dd"] > 0 and p.balance_dd > st["max_balance_dd"]:
            continue
        kept.append(p)

    key, desc = RANKERS[st["rank_by"]]
    kept.sort(key=key, reverse=desc)
    info["passed_constraints"] = len(kept)
    info["seconds"] = round(time.time() - t0, 1)
    if verbose:
        print(f"  {len(kept):,} portfolios pass constraints "
              f"({info['seconds']} s)")
    return kept[: st["keep_top"]], info


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export(portfolios, strategies, out_dir, name, export_top=3, info=None,
           filters=None, settings=None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = [app.save_portfolios(name, portfolios)]    # visible in the UI

    summary = {
        "name": name,
        "info": info or {},
        "filters": {**DEFAULT_FILTERS, **(filters or {})},
        "settings": {**DEFAULT_SETTINGS, **(settings or {})},
        "strategies": [s.name for s in strategies],
        "portfolios": [app.portfolio_to_dict(p) for p in portfolios],
    }
    path = out / f"portfolios_{name}.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    written.append(str(path))

    for k in range(min(export_top, len(portfolios))):
        html = app.generate_html_report(portfolios, strategies, selected_idx=k)
        p = out / f"portfolio_{k + 1}.html"
        p.write_text(html, encoding="utf-8")
        written.append(str(p))
        xlsx = app.generate_excel_report(portfolios, strategies, selected_idx=k)
        p = out / f"portfolio_{k + 1}.xlsx"
        p.write_bytes(xlsx)
        written.append(str(p))
    return written


def print_table(portfolios, top=10):
    print(f"\n{'#':>3} {'ret/DD':>7} {'profit':>11} {'eqDD':>9} {'corr':>6} "
          f"{'sharpe':>7} {'stab':>5} strategies")
    for i, p in enumerate(portfolios[:top], 1):
        names = ", ".join(n[:28] for n in p.strategy_names)
        print(f"{i:>3} {p.return_dd_ratio:7.2f} {p.total_profit:11,.0f} "
              f"{p.equity_dd:9,.0f} {p.avg_correlation:6.2f} {p.sharpe:7.2f} "
              f"{p.stability:5.2f} {names}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def run(html_dirs=(), databanks=(), lock_names=(), filters=None, settings=None,
        name=None, out_dir=None, export_top=3, save_databank=None,
        workers=None, verbose=True):
    """Library entry point; returns (portfolios, strategies, info, files)."""
    name = name or f"headless_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir = out_dir or str(HERE / "reports" / name)
    if verbose:
        print(f"Loading strategies...")
    strategies, locked = load_strategies(html_dirs, databanks, lock_names)
    if not strategies:
        raise SystemExit("No strategies loaded.")
    if save_databank:
        p = app.save_databank(save_databank, strategies, locked)
        if verbose:
            print(f"  combined set saved as databank: {p}")
    if verbose:
        print("Searching portfolios...")
    portfolios, info = generate_portfolios(strategies, locked, filters, settings,
                                           workers=workers, verbose=verbose)
    if not portfolios:
        print("No portfolios found. Filter rejections:")
        for k, v in info["filter_report"].items():
            print(f"    {k}: {v}")
        return [], strategies, info, []
    files = export(portfolios, strategies, out_dir, name, export_top, info,
                   filters, settings)
    if verbose:
        print_table(portfolios)
        print("\nWritten:")
        for f in files:
            print(f"  {f}")
    return portfolios, strategies, info, files


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--html", action="append", default=[], metavar="DIR")
    ap.add_argument("--databank", action="append", default=[], metavar="NAME")
    ap.add_argument("--lock", action="append", default=[], metavar="NAME")
    ap.add_argument("--name", help="run name (saved_portfolios/<name>.json)")
    ap.add_argument("--out", help="output folder (default reports/<name>)")
    ap.add_argument("--export-top", type=int, default=3)
    ap.add_argument("--save-databank", metavar="NAME",
                    help="also save the combined strategy set as a databank")
    ap.add_argument("--workers", type=int)
    # filters
    ap.add_argument("--min-trades", type=int, default=DEFAULT_FILTERS["min_trades"])
    ap.add_argument("--min-pf", type=float, default=DEFAULT_FILTERS["min_pf"])
    ap.add_argument("--min-sharpe", type=float, default=DEFAULT_FILTERS["min_sharpe"])
    ap.add_argument("--max-dd-pct", type=float, default=DEFAULT_FILTERS["max_dd_pct"])
    ap.add_argument("--min-profit", type=float)
    # settings
    ap.add_argument("--min-strategies", type=int, default=DEFAULT_SETTINGS["min_strategies"])
    ap.add_argument("--max-strategies", type=int, default=DEFAULT_SETTINGS["max_strategies"])
    ap.add_argument("--max-per-symbol", type=int, default=DEFAULT_SETTINGS["max_per_symbol"])
    ap.add_argument("--max-correlation", type=float, default=DEFAULT_SETTINGS["max_correlation"])
    ap.add_argument("--max-equity-dd", type=float, default=0.0)
    ap.add_argument("--max-balance-dd", type=float, default=0.0)
    ap.add_argument("--rank-by", choices=list(RANKERS), default="Return/DD")
    ap.add_argument("--keep-top", type=int, default=DEFAULT_SETTINGS["keep_top"])
    ap.add_argument("--max-iterations", type=int, default=DEFAULT_SETTINGS["max_iterations"])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    if not a.html and not a.databank:
        ap.error("give at least one --html DIR or --databank NAME")
    filters = dict(min_trades=a.min_trades, min_pf=a.min_pf, min_sharpe=a.min_sharpe,
                   max_dd_pct=a.max_dd_pct, min_profit=a.min_profit)
    settings = dict(min_strategies=a.min_strategies, max_strategies=a.max_strategies,
                    max_per_symbol=a.max_per_symbol, max_correlation=a.max_correlation,
                    max_equity_dd=a.max_equity_dd, max_balance_dd=a.max_balance_dd,
                    rank_by=a.rank_by, keep_top=a.keep_top,
                    max_iterations=a.max_iterations, seed=a.seed)
    run(a.html, a.databank, a.lock, filters, settings, a.name, a.out,
        a.export_top, a.save_databank, a.workers)


if __name__ == "__main__":
    main()

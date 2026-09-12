"""
Strategy bank: the on-disk home of strategies that PASSED the gate.

Layout
------
    strategy_bank/
        _runs.csv                 one line per test attempt (pass or fail);
                                  needed to count how many things were tried
        <folder>/<id>.json        one file per passing strategy

Each entry is a JSON document:
    {
      "id": "...", "tier": "A" | "B", "folder": "...", "saved_at": "...",
      "strategy": {...databank-format strategy: name, symbol, metrics, trades...},
      "validation": {...p-values, walk-forward, cross-market, prop-firm results...},
      "provenance": {...data, date ranges, instrument, rules, thresholds, versions...}
    }

The "strategy" block uses exactly the portfolio optimizer's databank format,
so any set of entries can be written as a databank the optimizer (UI or
headless.py) loads with send_to_optimizer().

Tier A = Python strategy that went through the full gate (validated edge).
Tier B = MT5 report only: prop-firm simulation and Monte Carlo on its trade
         list, no permutation / walk-forward tests (unvalidated edge).
"""

import csv
import json
import os
import re
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BANK_DIR = ROOT / "strategy_bank"
OPT_DIR = ROOT / "portfolio_optimizer"
if str(OPT_DIR) not in sys.path:
    sys.path.insert(0, str(OPT_DIR))

from mt5_parser import Strategy, StrategyMetrics, Trade   # noqa: E402


# ---------------------------------------------------------------------------
# Databank-format (de)serialization -- identical to the optimizer's
# ---------------------------------------------------------------------------

_METRIC_FIELDS = ["total_net_profit", "gross_profit", "gross_loss", "profit_factor",
                  "expected_payoff", "recovery_factor", "sharpe_ratio",
                  "balance_dd_absolute", "balance_dd_maximal", "balance_dd_maximal_pct",
                  "equity_dd_absolute", "equity_dd_maximal", "equity_dd_maximal_pct",
                  "total_trades", "profit_trades", "loss_trades", "win_rate",
                  "z_score", "lr_correlation", "initial_deposit"]


def strategy_to_dict(s):
    m = s.metrics
    return {
        "name": s.name, "symbol": s.symbol, "timeframe": s.timeframe,
        "period_start": s.period_start.isoformat() if s.period_start else None,
        "period_end": s.period_end.isoformat() if s.period_end else None,
        "file_path": s.file_path, "is_locked": s.is_locked,
        "metrics": {k: _finite(getattr(m, k)) for k in _METRIC_FIELDS},
        "trades": [{"time": t.time.isoformat(), "deal_id": t.deal_id, "symbol": t.symbol,
                    "trade_type": t.trade_type, "direction": t.direction,
                    "volume": t.volume, "price": t.price, "profit": t.profit,
                    "balance": t.balance, "comment": t.comment} for t in s.trades],
    }


def _finite(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return v
    if f != f or f in (float("inf"), float("-inf")):
        return 0.0
    return int(v) if isinstance(v, int) else f


def dict_to_strategy(d):
    metrics = StrategyMetrics(**{k: d["metrics"].get(k, 0) for k in _METRIC_FIELDS})
    trades = [Trade(time=datetime.fromisoformat(t["time"]), deal_id=t["deal_id"],
                    symbol=t["symbol"], trade_type=t["trade_type"],
                    direction=t["direction"], volume=t["volume"], price=t["price"],
                    profit=t["profit"], balance=t["balance"], comment=t.get("comment", ""))
              for t in d["trades"]]
    return Strategy(name=d["name"], symbol=d["symbol"], timeframe=d["timeframe"],
                    period_start=datetime.fromisoformat(d["period_start"]) if d.get("period_start") else None,
                    period_end=datetime.fromisoformat(d["period_end"]) if d.get("period_end") else None,
                    metrics=metrics, trades=trades, file_path=d.get("file_path", ""),
                    is_locked=d.get("is_locked", False))


def write_databank(path, strategies, locked_indices=()):
    """Write strategies in the optimizer's databank format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"saved_at": datetime.now().isoformat(), "num_strategies": len(strategies),
            "locked_indices": sorted(locked_indices),
            "strategies": [strategy_to_dict(s) for s in strategies]}
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return str(path)


def read_databank(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [dict_to_strategy(d) for d in data["strategies"]], set(data.get("locked_indices", []))


# ---------------------------------------------------------------------------
# Pass criteria
# ---------------------------------------------------------------------------

@dataclass
class Criteria:
    """Thresholds a strategy must clear to enter the bank. None = not checked."""
    max_is_p: Optional[float] = 0.01          # in-sample permutation p
    max_corrected_p: Optional[float] = 0.01   # multiple-system corrected p (when > 1 candidate)
    max_wf_p: Optional[float] = 0.05          # walk-forward permutation p
    min_wf_pf: Optional[float] = 1.0          # walk-forward OOS profit factor
    min_lockbox_pf: Optional[float] = 1.0     # lockbox profit factor (single look)
    min_p_pass: Optional[float] = 0.2         # prop-firm bootstrap P(pass)
    min_trades: Optional[int] = 30            # closed trades in the OOS period
    min_markets_passed: Optional[int] = None  # cross-market: markets that pass

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: v for k, v in (d or {}).items() if k in cls.__dataclass_fields__})


def passes(validation, criteria):
    """(passed, reasons) for a validation dict against Criteria."""
    v = validation or {}
    c = criteria
    reasons = []

    def chk(key, limit, higher_is_bad, label):
        if limit is None:
            return
        val = v.get(key)
        if val is None:
            reasons.append(f"{label}: missing")
            return
        bad = val > limit if higher_is_bad else val < limit
        if bad:
            reasons.append(f"{label} {val:.4g} {'>' if higher_is_bad else '<'} {limit}")

    chk("is_p", c.max_is_p, True, "in-sample p")
    if v.get("corrected_p") is not None or c.max_corrected_p is None:
        chk("corrected_p", c.max_corrected_p, True, "corrected p")
    chk("wf_p", c.max_wf_p, True, "walk-forward p")
    chk("wf_pf", c.min_wf_pf, False, "walk-forward PF")
    if v.get("lockbox_pf") is not None:
        chk("lockbox_pf", c.min_lockbox_pf, False, "lockbox PF")
    chk("p_pass", c.min_p_pass, False, "P(pass)")
    chk("trades", c.min_trades, False, "trades")
    chk("markets_passed", c.min_markets_passed, False, "markets passed")
    return (len(reasons) == 0), reasons


def passes_tier_b(validation, criteria):
    """Tier B only has prop-firm results: check P(pass) and trades."""
    v = validation or {}
    reasons = []
    if criteria.min_p_pass is not None and (v.get("p_pass") is None or v["p_pass"] < criteria.min_p_pass):
        reasons.append(f"P(pass) {v.get('p_pass', 'missing')} < {criteria.min_p_pass}")
    if criteria.min_trades is not None and (v.get("trades") or 0) < criteria.min_trades:
        reasons.append(f"trades {v.get('trades', 0)} < {criteria.min_trades}")
    return (len(reasons) == 0), reasons


# ---------------------------------------------------------------------------
# Folders and entries
# ---------------------------------------------------------------------------

def _safe(name):
    name = re.sub(r"[^A-Za-z0-9 _.\-\[\]]+", "_", str(name)).strip()
    return name or "unnamed"


def list_folders():
    BANK_DIR.mkdir(exist_ok=True)
    return sorted(p.name for p in BANK_DIR.iterdir() if p.is_dir() and not p.name.startswith("_"))


def create_folder(name):
    p = BANK_DIR / _safe(name)
    p.mkdir(parents=True, exist_ok=True)
    return p.name


def save_entry(folder, strategy, validation, provenance, tier="A", entry_id=None):
    """Save a passing strategy into strategy_bank/<folder>/<id>.json."""
    folder = create_folder(folder)
    entry_id = _safe(entry_id or f"{strategy.name}_{datetime.now():%Y%m%d_%H%M%S}")
    entry = {"id": entry_id, "tier": tier, "folder": folder,
             "saved_at": datetime.now().isoformat(),
             "strategy": strategy_to_dict(strategy),
             "validation": validation or {}, "provenance": provenance or {}}
    path = BANK_DIR / folder / f"{entry_id}.json"
    path.write_text(json.dumps(entry, indent=2, default=str), encoding="utf-8")
    return str(path)


def load_entry(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def list_entries(folder=None, with_trades=False):
    """
    Entries as light dicts (id, folder, tier, name, symbol, timeframe, saved_at,
    key validation numbers, path). with_trades=True returns the full documents.
    """
    BANK_DIR.mkdir(exist_ok=True)
    folders = [folder] if folder else list_folders()
    rows = []
    for f in folders:
        for p in sorted((BANK_DIR / f).glob("*.json")):
            e = load_entry(p)
            if with_trades:
                e["path"] = str(p)
                rows.append(e)
                continue
            s, v = e["strategy"], e.get("validation", {})
            rows.append({"id": e["id"], "folder": e["folder"], "tier": e["tier"],
                         "name": s["name"], "symbol": s["symbol"], "timeframe": s["timeframe"],
                         "saved_at": e["saved_at"], "path": str(p),
                         "trades": s["metrics"].get("total_trades"),
                         "net_profit": s["metrics"].get("total_net_profit"),
                         "profit_factor": s["metrics"].get("profit_factor"),
                         "equity_dd_pct": s["metrics"].get("equity_dd_maximal_pct"),
                         **{k: v.get(k) for k in ("is_p", "corrected_p", "wf_p", "wf_pf",
                                                  "lockbox_pf", "p_pass", "markets_passed",
                                                  "sharpe", "return_dd")}})
    return rows


def move_entry(path, new_folder):
    p = Path(path)
    e = load_entry(p)
    new_folder = create_folder(new_folder)
    e["folder"] = new_folder
    dest = BANK_DIR / new_folder / p.name
    dest.write_text(json.dumps(e, indent=2, default=str), encoding="utf-8")
    p.unlink()
    return str(dest)


def delete_entry(path):
    Path(path).unlink()


def entries_to_strategies(entries):
    """Full entry documents (or paths) -> optimizer Strategy objects."""
    out = []
    for e in entries:
        if isinstance(e, (str, Path)):
            e = load_entry(e)
        s = dict_to_strategy(e["strategy"])
        s.file_path = f"bank:{e['folder']}/{e['id']}"
        out.append(s)
    return out


def send_to_optimizer(entries, name, locked_ids=()):
    """
    Write bank entries as portfolio_optimizer/databanks/<name>.json so the
    optimizer UI (Load Databank) or headless.py (--databank NAME) can use them.
    Returns the databank path.
    """
    docs = [load_entry(e) if isinstance(e, (str, Path)) else e for e in entries]
    strategies = entries_to_strategies(docs)
    locked = {i for i, d in enumerate(docs) if d["id"] in set(locked_ids)}
    return write_databank(OPT_DIR / "databanks" / f"{_safe(name)}.json", strategies, locked)


# ---------------------------------------------------------------------------
# Run log (every attempt, pass or fail)
# ---------------------------------------------------------------------------

RUN_LOG = BANK_DIR / "_runs.csv"
_RUN_FIELDS = ["time", "strategy", "dataset", "timeframe", "tier", "is_p", "corrected_p",
               "wf_p", "wf_pf", "lockbox_pf", "p_pass", "trades", "passed", "reasons", "run_id"]


def log_run(record):
    BANK_DIR.mkdir(exist_ok=True)
    new = not RUN_LOG.exists()
    with open(RUN_LOG, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_RUN_FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow({**{k: "" for k in _RUN_FIELDS}, "time": datetime.now().isoformat(), **record})


def count_runs(dataset=None):
    """How many strategies have been tried (optionally on one dataset)."""
    if not RUN_LOG.exists():
        return 0
    with open(RUN_LOG, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if dataset:
        rows = [r for r in rows if r.get("dataset") == dataset]
    return len(rows)


if __name__ == "__main__":
    print("bank folder:", BANK_DIR)
    print("folders:", list_folders())
    for r in list_entries():
        print(f"  [{r['folder']}] {r['tier']} {r['name']}  p_pass={r['p_pass']}  wf_pf={r['wf_pf']}")
    print("runs logged:", count_runs())

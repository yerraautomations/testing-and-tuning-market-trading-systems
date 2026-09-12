"""
The pipeline: data -> every strategy through the gate -> multiple-system
correction -> criteria -> strategy bank -> (optionally) portfolio optimizer.

    python pipeline.py --config ../config/pipeline_example.json
    python pipeline.py --config my_run.json --no-optimizer

Non-negotiables built in (not settings):
  * every attempt is logged to strategy_bank/_runs.csv, pass or fail;
  * with two or more candidates the multiple-system permutation test runs
    over all of them and its corrected p-value is what the criteria check;
  * the lockbox is scored once per run and a warning is printed when the
    same strategy/dataset lockbox has been scored before.

Config keys (see config/pipeline_example.json):
  run_name, dataset, timeframe, instrument, rules, rules_overrides,
  strategies ("all" or list), extra_markets (list of dataset specs),
  lockbox_years, criteria, gate, bank_folder, mt5_reports (Tier B folders),
  optimizer ({"run": true, ...headless settings/filters...})
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import bank
from gate import (run_gate, run_tier_b, load_dataset, dataset_label, split_dev_lockbox,
                  DEFAULT_GATE)
from instruments import get_instrument
from propfirm import get_rules, rules_to_dict
from strategies_library import STRATEGIES, opt_for_insample, load_user_strategies
from mcpt_multi import multi_system_mcpt


def _lockbox_seen_before(strategy, dataset):
    if not bank.RUN_LOG.exists():
        return 0
    with open(bank.RUN_LOG, newline="", encoding="utf-8") as f:
        return sum(1 for r in csv.DictReader(f)
                   if r.get("strategy") == strategy and r.get("dataset") == dataset
                   and r.get("lockbox_pf") not in ("", "None", None))


def run_pipeline(config, verbose=True):
    say = print if verbose else (lambda *a, **k: None)
    t0 = time.time()
    load_user_strategies()

    run_name = config.get("run_name") or f"run_{datetime.now():%Y%m%d_%H%M%S}"
    run_id = f"{run_name}_{datetime.now():%H%M%S}"
    cfg = {**DEFAULT_GATE, **config.get("gate", {})}
    cfg["lockbox_years"] = config.get("lockbox_years", 0.0)
    timeframe = config.get("timeframe", "1D")
    instrument = get_instrument(config["instrument"])
    rules = get_rules(config["rules"], **config.get("rules_overrides", {}))
    criteria = bank.Criteria.from_dict(config.get("criteria"))
    folder = config.get("bank_folder") or f"auto/{run_name}"
    dataset_spec = dict(config["dataset"], timeframe=timeframe)
    label = dataset_label(dataset_spec)

    say(f"=== {run_name}: {label} | {instrument.symbol} | {rules.name} ===")
    df = load_dataset(dataset_spec)
    say(f"  {len(df):,} bars {df.index[0]:%Y-%m-%d} .. {df.index[-1]:%Y-%m-%d} UTC")
    extra = {}
    for spec in config.get("extra_markets", []):
        spec = dict(spec, timeframe=spec.get("timeframe", timeframe))
        extra[dataset_label(spec)] = load_dataset(spec)

    names = config.get("strategies", "all")
    if names == "all":
        names = list(STRATEGIES)
    results = {}
    for name in names:
        say(f"\n-- {name} --")
        try:
            results[name] = run_gate(name, df, instrument, rules, cfg, timeframe,
                                     extra_markets=extra or None, verbose=verbose)
        except Exception as e:      # one bad strategy must not kill the run
            say(f"  [{name}] ERROR: {e}")
            bank.log_run({"strategy": name, "dataset": label, "timeframe": timeframe,
                          "tier": "A", "passed": False, "reasons": f"error: {e}", "run_id": run_id})

    # multiple-system correction over everything tried in this run
    if len(results) >= 2:
        say(f"\n-- multiple-system correction over {len(results)} candidates --")
        dev, _ = split_dev_lockbox(df, cfg["lockbox_years"])
        train = dev.iloc[:int(len(dev) * cfg["train_frac"])].reset_index(drop=True)
        cands = {n: partial(opt_for_insample, optimize=STRATEGIES[n]["optimize"]) for n in results}
        ms = multi_system_mcpt(train, cands, n_permutations=cfg["is_perms"], seed=cfg["seed"],
                               n_jobs=cfg["n_jobs"], verbose=verbose)
        for i, n in enumerate(ms["names"]):
            results[n]["validation"]["corrected_p"] = float(ms["p_corrected"][i])
            results[n]["validation"]["candidates_in_run"] = len(results)
    else:
        for r in results.values():
            r["validation"]["corrected_p"] = r["validation"].get("is_p")
            r["validation"]["candidates_in_run"] = len(results)

    # criteria -> bank
    saved, rejected = [], []
    provenance_base = {"run_name": run_name, "run_id": run_id, "dataset": dataset_spec,
                       "instrument": instrument.symbol, "rules": rules_to_dict(rules),
                       "gate": cfg, "criteria": criteria.__dict__,
                       "extra_markets": list(extra), "when": datetime.now().isoformat(),
                       "runs_logged_before": bank.count_runs()}
    for name, r in results.items():
        v = r["validation"]
        if v.get("lockbox_pf") is not None and _lockbox_seen_before(name, label):
            say(f"  WARNING: lockbox for {name} on {label} was scored before "
                f"({_lockbox_seen_before(name, label)}x). Do not iterate on it.")
        ok, why = bank.passes(v, criteria)
        bank.log_run({"strategy": name, "dataset": label, "timeframe": timeframe, "tier": "A",
                      **{k: v.get(k) for k in ("is_p", "corrected_p", "wf_p", "wf_pf",
                                               "lockbox_pf", "p_pass", "trades")},
                      "passed": ok, "reasons": ";".join(why), "run_id": run_id})
        if ok:
            path = bank.save_entry(folder, r["strategy"], v, provenance_base, tier="A")
            saved.append(path)
            say(f"  PASS  {name} -> {path}")
        else:
            rejected.append((name, why))
            say(f"  fail  {name}: {'; '.join(why)}")

    # Tier B: MT5 reports
    for d in config.get("mt5_reports", []):
        d = d if os.path.isabs(d) else os.path.join(ROOT, d)
        for html in sorted(Path(d).glob("*.htm*")):
            try:
                r = run_tier_b(str(html), rules, cfg, verbose=verbose)
            except Exception as e:
                say(f"  [B] {html.name}: ERROR {e}")
                continue
            v = r["validation"]
            ok, why = bank.passes_tier_b(v, criteria)
            bank.log_run({"strategy": v["strategy"], "dataset": html.name, "tier": "B",
                          "p_pass": v["p_pass"], "trades": v["trades"], "passed": ok,
                          "reasons": ";".join(why), "run_id": run_id})
            if ok:
                path = bank.save_entry(folder, r["strategy"], v,
                                       {**provenance_base, "mt5_report": str(html)}, tier="B",
                                       entry_id=f"{r['strategy'].name}_{html.stem}")
                saved.append(path)
                say(f"  PASS  [B] {r['strategy'].name} -> {path}")
            else:
                rejected.append((f"[B] {r['strategy'].name}", why))

    summary = {"run_name": run_name, "run_id": run_id, "dataset": label,
               "instrument": instrument.symbol, "rules": rules.name, "bank_folder": folder,
               "tested": list(results), "saved": saved,
               "rejected": [{"strategy": n, "reasons": w} for n, w in rejected],
               "validation": {n: r["validation"] for n, r in results.items()},
               "seconds": round(time.time() - t0, 1)}

    # optimizer
    opt = config.get("optimizer", {})
    if opt.get("run", True) and len(saved) >= max(2, opt.get("min_strategies", 2)):
        say(f"\n-- portfolio optimizer on {len(saved)} bank entries --")
        db = bank.send_to_optimizer(saved, run_name)
        opt_dir = os.path.join(ROOT, "portfolio_optimizer")
        if opt_dir not in sys.path:
            sys.path.insert(0, opt_dir)
        import headless
        settings = {k: v for k, v in opt.items() if k in headless.DEFAULT_SETTINGS}
        filters = {k: v for k, v in opt.items() if k in headless.DEFAULT_FILTERS}
        filters.setdefault("min_trades", criteria.min_trades or 0)
        portfolios, _, info, files = headless.run(databanks=[run_name], name=run_name,
                                                  settings=settings, filters=filters,
                                                  export_top=opt.get("export_top", 3),
                                                  verbose=verbose)
        summary["optimizer"] = {"databank": db, "portfolios": len(portfolios),
                                "files": files, "info": info,
                                "top": [{"strategies": p.strategy_names,
                                         "return_dd": p.return_dd_ratio,
                                         "profit": p.total_profit, "equity_dd": p.equity_dd,
                                         "avg_correlation": p.avg_correlation}
                                        for p in portfolios[:5]]}
    elif opt.get("run", True):
        say(f"\n(optimizer skipped: {len(saved)} passing strategies)")

    out_dir = Path(ROOT) / "strategy_bank" / "_runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{run_id}.json"
    out.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    summary["summary_file"] = str(out)
    say(f"\nDone in {summary['seconds']} s. Bank entries saved: {len(saved)} "
        f"(tested {len(results)} Tier A strategies). Summary: {out}")
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="pipeline config JSON")
    ap.add_argument("--no-optimizer", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    with open(a.config, encoding="utf-8") as f:
        config = json.load(f)
    if a.no_optimizer:
        config.setdefault("optimizer", {})["run"] = False
    run_pipeline(config, verbose=not a.quiet)


if __name__ == "__main__":
    main()

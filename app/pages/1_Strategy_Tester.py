"""
Strategy Tester page: run the gate, manage the strategy bank, evaluate MT5
reports (Tier B), and send selections to the Portfolio Builder.
"""

import contextlib
import io
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
PY_DIR = ROOT / "python"
OPT_DIR = ROOT / "portfolio_optimizer"
for p in (PY_DIR, OPT_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

st.set_page_config(page_title="Strategy Tester", page_icon="🧪", layout="wide")

import bank                                                  # noqa: E402
import pipeline                                              # noqa: E402
from gate import run_tier_b, DEFAULT_GATE                    # noqa: E402
from instruments import load_profiles                        # noqa: E402
from propfirm import load_presets, get_rules                 # noqa: E402
from strategies_library import STRATEGIES, load_user_strategies   # noqa: E402

load_user_strategies()
PRESETS = load_presets()
PROFILES = load_profiles()
LAST_CFG = ROOT / "app" / "last_config.json"
TIMEFRAMES = ["1D", "4h", "1h", "15min", "5min", "native"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def dataset_choices():
    """Built-in names plus files in python/data/raw."""
    choices = {"BTC (built-in daily)": {"name": "BTC"}, "AAPL (built-in daily)": {"name": "AAPL"}}
    raw = PY_DIR / "data" / "raw"
    if raw.exists():
        for f in sorted(raw.glob("*.csv")):
            rel = f"data/raw/{f.name}"
            if "TICK" in f.name.upper():
                choices[f"{f.name} (ticks)"] = {"ticks": rel}
            else:
                choices[f"{f.name} (bars)"] = {"mt5": rel, "tz": "UTC"}
    return choices


def guess_instrument(label):
    up = label.upper()
    for sym in PROFILES:
        if sym.upper() in up:
            return sym
    return "EURUSD" if "USD" in up else list(PROFILES)[0]


def load_last():
    try:
        return json.loads(LAST_CFG.read_text(encoding="utf-8"))
    except Exception:
        return {}


def entries_frame(rows):
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    cols = ["select", "name", "folder", "tier", "symbol", "timeframe", "wf_pf", "is_p",
            "corrected_p", "wf_p", "lockbox_pf", "p_pass", "trades", "sharpe",
            "return_dd", "saved_at", "id", "path"]
    df.insert(0, "select", False)
    return df[[c for c in cols if c in df.columns]]


# ---------------------------------------------------------------------------
# page
# ---------------------------------------------------------------------------

st.title("🧪 Strategy Tester")
tab_run, tab_bank, tab_b = st.tabs(["Run the gate", "Strategy bank", "MT5 reports (Tier B)"])
last = load_last()

# ============================ Run the gate =================================
with tab_run:
    choices = dataset_choices()
    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.subheader("Data")
        ds_label = st.selectbox("Dataset", list(choices), index=max(0, list(choices).index(last.get("ds_label"))
                                if last.get("ds_label") in choices else 0))
        spec = dict(choices[ds_label])
        c1, c2, c3 = st.columns(3)
        timeframe = c1.selectbox("Timeframe", TIMEFRAMES, index=TIMEFRAMES.index(last.get("timeframe", "1D")))
        if "mt5" in spec:
            spec["tz"] = c2.selectbox("File timezone", ["UTC", "broker"], help="broker = GMT+2/+3 with US DST (MT5 server time)")
        since = c3.text_input("Since (YYYY-MM-DD, optional)", last.get("since", ""))
        if since.strip():
            spec["since"] = since.strip()
        lockbox_years = st.number_input("Lockbox (years at the end, scored once)", 0.0, 10.0,
                                        float(last.get("lockbox_years", 2.0)), 0.5)
    with col_b:
        st.subheader("Instrument & firm")
        inst_names = list(PROFILES)
        default_inst = last.get("instrument") or guess_instrument(ds_label)
        instrument = st.selectbox("Instrument profile", inst_names,
                                  index=inst_names.index(default_inst) if default_inst in inst_names else 0)
        preset_names = list(PRESETS)
        preset = st.selectbox("Prop-firm rules", preset_names,
                              index=preset_names.index(last.get("rules", "FTMO 2-Step")) if last.get("rules") in preset_names else 0)
        base = PRESETS[preset]
        if not base.verified:
            st.caption("⚠️ preset not verified against the firm's current terms — check the numbers")

    with st.expander("Edit the rules (overrides the preset for this run)"):
        r1, r2, r3, r4 = st.columns(4)
        targets_txt = r1.text_input("Profit targets (fractions, comma-separated; empty = none)",
                                    ", ".join(str(t) for t in base.profit_targets))
        max_daily = r2.number_input("Max daily loss", 0.0, 1.0, float(base.max_daily_loss), 0.005, format="%.3f")
        max_total = r3.number_input("Max total loss", 0.0, 1.0, float(base.max_total_loss), 0.005, format="%.3f")
        account = r4.number_input("Account size", 1000.0, 10_000_000.0, float(base.account_size), 1000.0)
        r5, r6, r7, r8 = st.columns(4)
        mode = r5.selectbox("Max loss mode", ["static", "trailing"], index=0 if base.max_loss_mode == "static" else 1)
        basis = r6.selectbox("Trailing basis", ["eod", "intraday"], index=0 if base.trailing_basis == "eod" else 1)
        lock = r7.checkbox("Trailing locks at initial balance", base.trailing_lock_at_initial)
        min_days = r8.number_input("Min trading days", 0, 60, int(base.min_trading_days))
        r9, r10, r11, r12 = st.columns(4)
        consistency = r9.number_input("Consistency: max best-day share (0 = none)", 0.0, 1.0,
                                      float(base.consistency_max_day_pct or 0.0), 0.05)
        payout = r10.number_input("Payout threshold (no-target accounts)", 0.0, 1.0, float(base.payout_threshold), 0.01)
        fee = r11.number_input("Fee", 0.0, 100_000.0, float(base.fee), 10.0)
        tz = r12.text_input("Daily reset timezone", base.daily_reset_tz)
    overrides = dict(
        profit_targets=tuple(float(x) for x in targets_txt.split(",") if x.strip()),
        max_daily_loss=max_daily, max_total_loss=max_total, account_size=account,
        max_loss_mode=mode, trailing_basis=basis, trailing_lock_at_initial=lock,
        min_trading_days=int(min_days), consistency_max_day_pct=(consistency or None),
        payout_threshold=payout, fee=fee, daily_reset_tz=tz)

    st.subheader("Strategies")
    strat_names = list(STRATEGIES)
    chosen = st.multiselect("Strategies to run through the gate", strat_names,
                            default=[s for s in last.get("strategies", strat_names) if s in strat_names])

    with st.expander("Pass criteria (a strategy enters the bank only if it clears all of these)"):
        cdef = bank.Criteria()
        k1, k2, k3, k4 = st.columns(4)
        max_is_p = k1.number_input("Max in-sample p", 0.0, 1.0, cdef.max_is_p, 0.005, format="%.3f")
        max_corr_p = k2.number_input("Max corrected p", 0.0, 1.0, cdef.max_corrected_p, 0.005, format="%.3f")
        max_wf_p = k3.number_input("Max walk-forward p", 0.0, 1.0, cdef.max_wf_p, 0.005, format="%.3f")
        min_wf_pf = k4.number_input("Min walk-forward PF", 0.0, 5.0, cdef.min_wf_pf, 0.05)
        k5, k6, k7, k8 = st.columns(4)
        min_lock_pf = k5.number_input("Min lockbox PF", 0.0, 5.0, cdef.min_lockbox_pf, 0.05)
        min_p_pass = k6.number_input("Min P(pass)", 0.0, 1.0, cdef.min_p_pass, 0.05)
        min_trades = k7.number_input("Min trades", 0, 10000, cdef.min_trades)
        min_mkts = k8.number_input("Min markets passed (0 = ignore)", 0, 20, 0)
    criteria = dict(max_is_p=max_is_p, max_corrected_p=max_corr_p, max_wf_p=max_wf_p,
                    min_wf_pf=min_wf_pf, min_lockbox_pf=min_lock_pf, min_p_pass=min_p_pass,
                    min_trades=int(min_trades), min_markets_passed=(int(min_mkts) or None))

    with st.expander("Gate settings"):
        g1, g2, g3, g4 = st.columns(4)
        is_perms = g1.number_input("In-sample permutations", 20, 5000, int(DEFAULT_GATE["is_perms"]), 10)
        wf_perms = g2.number_input("Walk-forward permutations", 20, 5000, int(DEFAULT_GATE["wf_perms"]), 10)
        n_sims = g3.number_input("Prop-firm simulations", 100, 20000, int(DEFAULT_GATE["n_sims"]), 100)
        stress = g4.number_input("Cost stress multiplier", 0.5, 5.0, float(DEFAULT_GATE["cost_stress"]), 0.5)
        g5, g6, g7, g8 = st.columns(4)
        train_bars = g5.number_input("Walk-forward train bars", 50, 100000, int(DEFAULT_GATE["train_bars"]), 50)
        train_step = g6.number_input("Re-optimize every (bars)", 10, 20000, int(DEFAULT_GATE["train_step"]), 10)
        sigmas = g7.number_input("Vol target: daily-loss sigmas", 1.0, 10.0, float(DEFAULT_GATE["target_vol_sigmas"]), 0.5)
        horizon = g8.number_input("Challenge horizon (days)", 30, 2000, int(DEFAULT_GATE["horizon"]), 30)
    gate_cfg = dict(is_perms=int(is_perms), wf_perms=int(wf_perms), n_sims=int(n_sims), cost_stress=stress,
                    train_bars=int(train_bars), train_step=int(train_step), target_vol_sigmas=sigmas,
                    horizon=int(horizon))

    st.subheader("Bank folder for passers")
    folders = bank.list_folders()
    f1, f2 = st.columns(2)
    folder_pick = f1.selectbox("Existing folder", ["(new)"] + folders)
    folder_new = f2.text_input("New folder name", last.get("bank_folder", "candidates"))
    folder = folder_new.strip() if folder_pick == "(new)" else folder_pick
    run_name = st.text_input("Run name", f"{ds_label.split(' ')[0]}_{timeframe}_{time.strftime('%Y%m%d')}")

    if st.button("▶ Run the gate", type="primary", disabled=not chosen):
        config = {"run_name": run_name, "dataset": spec, "timeframe": timeframe,
                  "instrument": instrument, "rules": preset, "rules_overrides": overrides,
                  "strategies": chosen, "lockbox_years": lockbox_years, "criteria": criteria,
                  "gate": gate_cfg, "bank_folder": folder, "optimizer": {"run": False}}
        LAST_CFG.write_text(json.dumps({"ds_label": ds_label, "timeframe": timeframe, "since": since,
                                        "lockbox_years": lockbox_years, "instrument": instrument,
                                        "rules": preset, "strategies": chosen, "bank_folder": folder},
                                       indent=2), encoding="utf-8")
        log = io.StringIO()
        with st.status("Running the gate… this can take several minutes", expanded=True) as status:
            try:
                with contextlib.redirect_stdout(log):
                    summary = pipeline.run_pipeline(config, verbose=True)
                status.update(label="Gate finished", state="complete")
            except Exception as e:
                status.update(label=f"Gate failed: {e}", state="error")
                summary = None
        st.session_state["last_summary"] = summary
        st.session_state["last_log"] = log.getvalue()

    summary = st.session_state.get("last_summary")
    if summary:
        st.subheader(f"Results: {summary['run_name']}")
        rows = []
        rejected = {r["strategy"]: r["reasons"] for r in summary["rejected"]}
        for name, v in summary["validation"].items():
            rows.append({"strategy": name, "passed": name not in rejected,
                         "is_pf": v.get("is_pf"), "is_p": v.get("is_p"), "corrected_p": v.get("corrected_p"),
                         "wf_pf": v.get("wf_pf"), "wf_p": v.get("wf_p"), "lockbox_pf": v.get("lockbox_pf"),
                         "p_pass": v.get("p_pass"), "fail_daily": v.get("p_fail_daily"),
                         "fail_max": v.get("p_fail_max"), "median_days": v.get("median_days"),
                         "trades": v.get("trades"), "sharpe": v.get("sharpe"),
                         "why rejected": "; ".join(rejected.get(name, []))})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.write(f"Saved to bank folder **{summary['bank_folder']}**: {len(summary['saved'])} strategies")
        with st.expander("Run log"):
            st.code(st.session_state.get("last_log", ""))

# ============================ Strategy bank ================================
with tab_bank:
    st.subheader("Strategy bank")
    b1, b2 = st.columns([3, 1])
    with b2:
        new_folder = st.text_input("Create folder")
        if st.button("Create") and new_folder.strip():
            bank.create_folder(new_folder.strip())
            st.rerun()
    folders = bank.list_folders()
    with b1:
        pick = st.multiselect("Folders to show", folders, default=folders)
    rows = [r for f in pick for r in bank.list_entries(f)]
    df = entries_frame(rows)
    if df.empty:
        st.info("No strategies in the selected folders yet. Passers from the gate appear here.")
    else:
        sort_by = st.selectbox("Rank by", [c for c in ["wf_pf", "p_pass", "return_dd", "sharpe", "corrected_p",
                                                       "wf_p", "lockbox_pf", "trades"] if c in df.columns])
        df = df.sort_values(sort_by, ascending=sort_by in ("corrected_p", "wf_p"), na_position="last")
        edited = st.data_editor(df, use_container_width=True, hide_index=True,
                                disabled=[c for c in df.columns if c != "select"],
                                column_config={"path": None})
        selected = edited[edited["select"]]
        st.caption(f"{len(selected)} selected")
        a1, a2, a3, a4, a5 = st.columns(5)
        target = a1.selectbox("Move to folder", ["(choose)"] + folders)
        if a1.button("Move", disabled=selected.empty or target == "(choose)"):
            for p in selected["path"]:
                bank.move_entry(p, target)
            st.rerun()
        if a2.button("Delete selected", disabled=selected.empty):
            for p in selected["path"]:
                bank.delete_entry(p)
            st.rerun()
        db_name = a3.text_input("Databank name", f"bank_{time.strftime('%Y%m%d_%H%M')}")
        if a3.button("Save as databank", disabled=selected.empty):
            path = bank.send_to_optimizer(list(selected["path"]), db_name)
            st.success(f"Saved {path}. Load it in the Portfolio Builder sidebar.")
        if a4.button("➡ Send to Portfolio Builder", type="primary", disabled=selected.empty):
            strategies = bank.entries_to_strategies(list(selected["path"]))
            st.session_state["strategies"] = strategies
            st.session_state["locked_indices"] = set()
            st.session_state["portfolios"] = []
            st.session_state["selected_portfolio"] = None
            bank.send_to_optimizer(list(selected["path"]), db_name)
            st.switch_page("pages/2_Portfolio_Builder.py")
        if a5.button("Run headless optimizer", disabled=selected.empty,
                     help="Command-line portfolio search on the selection; reports in portfolio_optimizer/reports"):
            path = bank.send_to_optimizer(list(selected["path"]), db_name)
            cmd = [sys.executable, str(OPT_DIR / "headless.py"), "--databank", db_name,
                   "--name", db_name, "--min-trades", "1", "--max-per-symbol", "10"]
            with st.spinner("Searching portfolios…"):
                out = subprocess.run(cmd, capture_output=True, text=True, cwd=str(OPT_DIR))
            st.code(out.stdout[-6000:] + ("\n" + out.stderr[-3000:] if out.returncode else ""))

# ============================ Tier B =======================================
with tab_b:
    st.subheader("MT5 reports → prop-firm simulation only (Tier B)")
    st.caption("An EA that exists only as an MT5 backtest cannot go through the permutation and walk-forward "
               "tests. Its trade list is simulated against the prop-firm rules and saved as an *unvalidated* "
               "strategy you can still use in portfolios.")
    files = st.file_uploader("MT5 Strategy Tester HTML reports", type=["html", "htm"], accept_multiple_files=True)
    tb1, tb2, tb3 = st.columns(3)
    preset_b = tb1.selectbox("Prop-firm rules", list(PRESETS), key="tier_b_rules")
    min_p_pass_b = tb2.number_input("Min P(pass) to keep", 0.0, 1.0, 0.2, 0.05, key="tier_b_min")
    folder_b = tb3.text_input("Bank folder", "mt5_reports")
    if files and st.button("Evaluate reports"):
        rules = get_rules(preset_b)
        up = bank.BANK_DIR / "_uploads"
        up.mkdir(parents=True, exist_ok=True)
        results = []
        for f in files:
            path = up / f.name
            path.write_bytes(f.getvalue())
            try:
                r = run_tier_b(str(path), rules, verbose=False)
            except Exception as e:
                st.warning(f"{f.name}: {e}")
                continue
            v = r["validation"]
            keep = v["p_pass"] >= min_p_pass_b
            results.append({"file": f.name, "strategy": r["strategy"].name, "trades": v["trades"],
                            "PF": v["is_pf"], "P(pass)": v["p_pass"], "fail daily": v["p_fail_daily"],
                            "fail max": v["p_fail_max"], "kept": keep})
            bank.log_run({"strategy": r["strategy"].name, "dataset": f.name, "tier": "B",
                          "p_pass": v["p_pass"], "trades": v["trades"], "passed": keep,
                          "reasons": "" if keep else f"P(pass) < {min_p_pass_b}", "run_id": "ui"})
            if keep:
                bank.save_entry(folder_b, r["strategy"], v, {"mt5_report": f.name, "rules": rules.name,
                                                             "tier": "B", "when": time.strftime("%Y-%m-%dT%H:%M:%S")},
                                tier="B", entry_id=f"{r['strategy'].name}_{Path(f.name).stem}")
        st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)

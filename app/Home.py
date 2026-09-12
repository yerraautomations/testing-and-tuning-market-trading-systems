"""
Trading Workbench -- entry page.

    streamlit run app/Home.py        (or double-click Launch_App.bat)

Pages (left sidebar):
  1. Strategy Tester   run the honesty gate, keep the passers in the strategy bank
  2. Portfolio Builder the portfolio optimizer (unchanged), fed from the bank
"""

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "python", ROOT / "portfolio_optimizer"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

st.set_page_config(page_title="Trading Workbench", page_icon="🧭", layout="wide")

import bank                                   # noqa: E402
from propfirm import load_presets             # noqa: E402
from instruments import load_profiles         # noqa: E402
from strategies_library import STRATEGIES, load_user_strategies   # noqa: E402

load_user_strategies()

st.title("🧭 Trading Workbench")
st.caption("Validate strategies honestly, keep the survivors, build portfolios from them.")

c1, c2, c3, c4 = st.columns(4)
entries = bank.list_entries()
c1.metric("Strategies in bank", len(entries))
c2.metric("Bank folders", len(bank.list_folders()))
c3.metric("Registered strategies", len(STRATEGIES))
c4.metric("Test runs logged", bank.count_runs())

st.markdown("""
### How it flows
1. **Strategy Tester** — pick data, an instrument profile, a prop-firm rule set and the strategies to test.
   The gate runs: in-sample permutation test → multiple-system correction → walk-forward →
   walk-forward permutation test → lockbox (scored once) → prop-firm simulation.
   Only strategies that clear your criteria are saved, into the bank folder you choose.
2. **Strategy bank** — folders of validated strategies (Tier A) and MT5-report strategies (Tier B,
   prop-firm simulation only). Rank, move, delete, or send a selection to the builder.
3. **Portfolio Builder** — the optimizer: correlations, combinations, locked strategies, Monte Carlo, reports.
""")

with st.expander("Available prop-firm presets"):
    rows = [{"preset": r.name, "targets": r.profit_targets, "daily loss": r.max_daily_loss,
             "max loss": r.max_total_loss, "mode": r.max_loss_mode, "verified": r.verified}
            for r in load_presets().values()]
    st.dataframe(rows, use_container_width=True, hide_index=True)
with st.expander("Instrument profiles"):
    st.dataframe([{"symbol": p.symbol, "class": p.asset_class, "tick": p.tick_size,
                   "multiplier": p.multiplier, "spread/slippage": p.spread_ticks,
                   "commission": p.commission} for p in load_profiles().values()],
                 use_container_width=True, hide_index=True)
with st.expander("Registered strategies"):
    st.dataframe([{"name": n, "kind": s.get("kind", ""), "params": s.get("params", "")}
                  for n, s in STRATEGIES.items()], use_container_width=True, hide_index=True)

st.info("Use the sidebar to open the Strategy Tester or the Portfolio Builder.")

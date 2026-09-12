"""
Portfolio Builder page: runs portfolio_optimizer/streamlit_app.py exactly as
it is, inside the workbench. Nothing in the optimizer is modified.

Strategies sent from the Strategy Tester page arrive through
st.session_state["strategies"], which the optimizer keeps because its
init_session_state() only fills keys that are missing.
"""

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OPT_DIR = ROOT / "portfolio_optimizer"
if str(OPT_DIR) not in sys.path:
    sys.path.insert(0, str(OPT_DIR))

runpy.run_path(str(OPT_DIR / "streamlit_app.py"), run_name="__main__")

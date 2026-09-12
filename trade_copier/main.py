#!/usr/bin/env python3
"""
Trade Copier -- entry point.

    python main.py --test              # check MT5 and every slave connection
    python main.py --dry-run           # run the engine, log what it WOULD do, touch nothing
    python main.py                     # copy for real
    python main.py --headed            # show the slave browser window (debugging)
    python main.py --config other.yaml

Configuration: config/settings.yaml (template, committed) overridden key by
key by config/settings.local.yaml (your real values, git-ignored), then by
environment variables TC_SLAVE_EMAIL, TC_SLAVE_PASSWORD (applied to every
slave that has no password) and TC_MT5_ACCOUNT / TC_MT5_PASSWORD / TC_MT5_SERVER.
"""

import argparse
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(SCRIPT_DIR))

import yaml   # noqa: E402

from utils.logger import setup_logger                      # noqa: E402
from utils.notifier import Notifier                        # noqa: E402
from core.copier_engine import CopierEngine, SlaveConfig   # noqa: E402


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

def _merge(base, over):
    """Deep merge; lists of slaves are merged by 'name'."""
    if isinstance(base, dict) and isinstance(over, dict):
        out = dict(base)
        for k, v in over.items():
            out[k] = _merge(base.get(k), v) if k in base else v
        return out
    if isinstance(base, list) and isinstance(over, list) and base and isinstance(base[0], dict) and "name" in base[0]:
        by_name = {d.get("name"): d for d in base}
        for d in over:
            n = d.get("name")
            by_name[n] = _merge(by_name[n], d) if n in by_name else d
        return list(by_name.values())
    return over if over is not None else base


def load_config(path):
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    local = path.with_name(path.stem + ".local" + path.suffix)
    if local.exists():
        with open(local, encoding="utf-8") as f:
            cfg = _merge(cfg, yaml.safe_load(f) or {})
    m = cfg.setdefault("master", {})
    for key, env in (("account", "TC_MT5_ACCOUNT"), ("password", "TC_MT5_PASSWORD"), ("server", "TC_MT5_SERVER")):
        if os.environ.get(env):
            m[key] = int(os.environ[env]) if key == "account" else os.environ[env]
    for s in cfg.get("slaves", []):
        if not s.get("password") and os.environ.get("TC_SLAVE_PASSWORD"):
            s["password"] = os.environ["TC_SLAVE_PASSWORD"]
        if not s.get("email") and os.environ.get("TC_SLAVE_EMAIL"):
            s["email"] = os.environ["TC_SLAVE_EMAIL"]
    return cfg


# ---------------------------------------------------------------------------
# factories
# ---------------------------------------------------------------------------

def create_master(cfg):
    from forex.mt5_monitor import MT5Monitor
    m = cfg.get("master", {})
    return MT5Monitor(terminal_path=m.get("terminal_path"), account=m.get("account"),
                      password=m.get("password"), server=m.get("server"))


def create_adapter(slave, headless=True):
    platform = (slave.get("platform") or "").lower()
    if platform == "fake":
        from core.fake_adapter import FakeAdapter
        return FakeAdapter(slave.get("name", "fake"))
    if platform == "matchtrader":
        if slave.get("use_browser", True):
            try:
                from forex.matchtrader_web import MatchTraderWeb
            except ImportError as e:
                raise SystemExit(
                    "forex/matchtrader_web.py is not available yet. Record a session with\n"
                    "  python tools/record_session.py --url <platform url>\n"
                    f"so the platform's calls can be implemented. ({e})")
            return MatchTraderWeb(server_url=slave["server"], email=slave.get("email", ""),
                                  password=slave.get("password", ""),
                                  account_number=slave.get("account_number"), headless=headless)
        from forex.matchtrader_adapter import MatchTraderAdapter
        return MatchTraderAdapter(server_url=slave["server"], email=slave.get("email", ""),
                                  password=slave.get("password", ""),
                                  account_number=slave.get("account_number"))
    raise SystemExit(f"Unknown slave platform: {platform!r}")


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------

def test_connections(cfg, log):
    ok = True
    log.info("Testing MT5 (master)...")
    master = create_master(cfg)
    if master.initialize():
        info = master.get_account_info()
        log.info(f"  MT5 OK: account {info['login']} on {info['server']}, balance {info['balance']:.2f} {info['currency']}")
        log.info(f"  open positions: {len(master.get_positions())}")
        master.shutdown()
    else:
        log.error("  MT5 connection FAILED (is the terminal running and logged in?)")
        ok = False
    for s in cfg.get("slaves", []):
        if not s.get("enabled", True):
            continue
        name = s.get("name", "slave")
        log.info(f"Testing {name} ({s.get('platform')})...")
        try:
            a = create_adapter(s, headless=False)
            if a.connect():
                info = a.get_account_info() or {}
                pos = a.get_positions()
                log.info(f"  {name} OK: balance {info.get('balance')}, equity {info.get('equity')}, "
                         f"{len(pos)} open positions")
                for p in pos:
                    log.info(f"    {p['id']} {p['symbol']} {p['side']} {p['volume']}")
                a.disconnect()
            else:
                log.error(f"  {name} connection FAILED")
                ok = False
        except SystemExit as e:
            log.error(f"  {name}: {e}")
            ok = False
        except Exception as e:
            log.error(f"  {name} error: {e}", exc_info=True)
            ok = False
    return ok


def run(cfg, log, dry_run=False, headless=True):
    master = create_master(cfg)
    if not master.initialize():
        log.error("Cannot connect to MT5. Exiting.")
        return False
    notif = cfg.get("notifications") or {}
    notifier = Notifier(notif.get("discord_webhook") or None, notif.get("telegram_bot_token") or None,
                        notif.get("telegram_chat_id") or None) if notif.get("enabled") else None
    engine = CopierEngine(master, state_path=SCRIPT_DIR / "state" / "positions.json",
                          poll_interval_ms=cfg.get("master", {}).get("poll_interval_ms", 250),
                          heartbeat_s=cfg.get("heartbeat_s", 30), dry_run=dry_run, notifier=notifier)
    for s in cfg.get("slaves", []):
        if not s.get("enabled", True):
            continue
        adapter = create_adapter(s, headless=headless)
        if adapter.connect():
            engine.add_slave(SlaveConfig.from_dict(s, adapter))
        else:
            log.error(f"{s.get('name')}: connection failed; not added")
    if not engine.slaves:
        log.error("No slaves connected. Exiting.")
        master.shutdown()
        return False
    info = master.get_account_info()
    log.info(f"Master: MT5 {info['login']} ({info['server']}), balance {info['balance']:.2f}")
    log.info(f"Slaves: {', '.join(engine.slaves)}{'  [DRY RUN]' if dry_run else ''}")
    try:
        engine.start()
    finally:
        for c in engine.slaves.values():
            try:
                c.adapter.disconnect()
            except Exception:
                pass
        master.shutdown()
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", "-c", default="config/settings.yaml")
    ap.add_argument("--test", "-t", action="store_true", help="test connections only")
    ap.add_argument("--dry-run", action="store_true", help="log actions, do not trade on slaves")
    ap.add_argument("--headed", action="store_true", help="show the slave browser window")
    ap.add_argument("--debug", "-d", action="store_true")
    a = ap.parse_args()

    path = Path(a.config)
    if not path.is_absolute():
        path = SCRIPT_DIR / path
    if not path.exists():
        raise SystemExit(f"Config not found: {path}")
    cfg = load_config(path)
    lg = cfg.get("logging") or {}
    log = setup_logger("trade_copier", "DEBUG" if a.debug else lg.get("level", "INFO"),
                       str(SCRIPT_DIR / lg.get("file", "logs/copier.log")),
                       lg.get("max_size_mb", 10), lg.get("backup_count", 5))
    log.info(f"Trade Copier 2.0  config={path}")
    if a.test:
        sys.exit(0 if test_connections(cfg, log) else 1)
    sys.exit(0 if run(cfg, log, dry_run=a.dry_run, headless=not a.headed) else 1)


if __name__ == "__main__":
    main()

"""Engine tests against the fake platform. Run: pytest trade_copier/tests -q"""

import sys
from datetime import datetime
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.copier_engine import CopierEngine, SlaveConfig      # noqa: E402
from core.fake_adapter import FakeAdapter                      # noqa: E402
from core.position import Position, PositionType               # noqa: E402


class FakeMaster:
    """Stand-in for the MT5 monitor: a list we mutate between polls."""
    def __init__(self):
        self.positions = []

    def get_positions(self):
        return [Position(**p) for p in self.positions]

    def open(self, ticket, symbol, side="BUY", volume=0.1, sl=0.0, tp=0.0, magic=0):
        self.positions.append(dict(ticket=ticket, symbol=symbol,
                                   position_type=PositionType[side], volume=volume,
                                   open_price=1.0, open_time=datetime.now(), stop_loss=sl,
                                   take_profit=tp, magic_number=magic))

    def close(self, ticket):
        self.positions = [p for p in self.positions if p["ticket"] != ticket]

    def reduce(self, ticket, new_volume):
        for p in self.positions:
            if p["ticket"] == ticket:
                p["volume"] = new_volume

    def modify(self, ticket, sl=None, tp=None):
        for p in self.positions:
            if p["ticket"] == ticket:
                if sl is not None:
                    p["stop_loss"] = sl
                if tp is not None:
                    p["take_profit"] = tp


def make(tmp_path, **cfg):
    master = FakeMaster()
    slave = FakeAdapter("IF", specs={"EURUSD.x": {"min_volume": 0.01, "volume_step": 0.01, "max_volume": 50}})
    slave.connect()
    engine = CopierEngine(master, state_path=tmp_path / "positions.json", heartbeat_s=9999)
    d = {"name": "IF", "platform": "fake", "symbol_suffix": ".x", "lot_multiplier": 0.5,
         "copy_sl": True, "copy_tp": True}
    d.update(cfg)
    engine.add_slave(SlaveConfig.from_dict(d, slave))
    engine.reconcile_on_start()
    return engine, master, slave


def test_open_is_copied_with_multiplier_suffix_and_link(tmp_path):
    engine, master, slave = make(tmp_path)
    master.open(1, "EURUSD", "BUY", 0.10, sl=1.09, tp=1.12)
    engine.poll_once()
    assert len(slave.positions) == 1
    p = next(iter(slave.positions.values()))
    assert p["symbol"] == "EURUSD.x" and p["side"] == "BUY" and p["volume"] == 0.05
    assert p["sl"] == 1.09 and p["tp"] == 1.12
    assert engine.state.slave_of(1, "IF")["id"] == p["id"]
    assert engine.stats["copied"] == 1


def test_close_propagates_and_link_is_forgotten(tmp_path):
    engine, master, slave = make(tmp_path)
    master.open(1, "EURUSD", "SELL", 0.10)
    engine.poll_once()
    master.close(1)
    engine.poll_once()
    assert slave.positions == {}
    assert engine.state.links == {}
    assert engine.stats["closed"] == 1


def test_partial_close_is_mirrored_proportionally(tmp_path):
    engine, master, slave = make(tmp_path, lot_multiplier=1.0)
    master.open(1, "EURUSD", "BUY", 0.10)
    engine.poll_once()
    master.reduce(1, 0.04)                  # closed 0.06 of 0.10
    engine.poll_once()
    p = next(iter(slave.positions.values()))
    assert abs(p["volume"] - 0.04) < 1e-9
    assert abs(engine.state.slave_of(1, "IF")["volume"] - 0.04) < 1e-9


def test_below_minimum_volume_is_not_copied(tmp_path):
    engine, master, slave = make(tmp_path, lot_multiplier=0.1)
    master.open(1, "EURUSD", "BUY", 0.01)   # 0.001 -> below 0.01 minimum
    engine.poll_once()
    assert slave.positions == {}
    assert engine.stats["skipped"] == 1


def test_volume_is_floored_to_step(tmp_path):
    engine, master, slave = make(tmp_path, lot_multiplier=0.33)
    master.open(1, "EURUSD", "BUY", 0.10)   # 0.033 -> 0.03
    engine.poll_once()
    assert next(iter(slave.positions.values()))["volume"] == 0.03


def test_restart_closes_slave_when_master_closed_while_down(tmp_path):
    engine, master, slave = make(tmp_path)
    master.open(1, "EURUSD", "BUY", 0.10)
    engine.poll_once()
    assert len(slave.positions) == 1
    # copier "dies", master closes meanwhile, copier restarts with the same state file
    master.close(1)
    engine2 = CopierEngine(master, state_path=tmp_path / "positions.json", heartbeat_s=9999)
    engine2.add_slave(SlaveConfig.from_dict({"name": "IF", "platform": "fake", "symbol_suffix": ".x"}, slave))
    engine2.reconcile_on_start()
    assert slave.positions == {}
    assert engine2.state.links == {}


def test_restart_keeps_links_for_still_open_master_positions(tmp_path):
    engine, master, slave = make(tmp_path)
    master.open(1, "EURUSD", "BUY", 0.10)
    engine.poll_once()
    sid = engine.state.slave_of(1, "IF")["id"]
    engine2 = CopierEngine(master, state_path=tmp_path / "positions.json", heartbeat_s=9999)
    engine2.add_slave(SlaveConfig.from_dict({"name": "IF", "platform": "fake", "symbol_suffix": ".x"}, slave))
    engine2.reconcile_on_start()
    assert engine2.state.slave_of(1, "IF")["id"] == sid
    master.close(1)
    engine2.poll_once()
    assert slave.positions == {}


def test_unknown_slave_position_is_reported_not_closed_by_default(tmp_path):
    engine, master, slave = make(tmp_path)
    orphan = slave.inject_position("GBPUSD.x", "BUY", 0.02)
    engine.check_unknown_slave_positions()
    assert orphan in slave.positions


def test_unknown_slave_position_closed_when_configured(tmp_path):
    engine, master, slave = make(tmp_path, close_unknown_slave_positions=True)
    orphan = slave.inject_position("GBPUSD.x", "BUY", 0.02)
    engine.check_unknown_slave_positions()
    assert orphan not in slave.positions


def test_open_failure_is_retried_then_alerted(tmp_path):
    engine, master, slave = make(tmp_path, max_retries=2)
    slave.fail_opens = 5
    master.open(1, "EURUSD", "BUY", 0.10)
    engine.poll_once()
    assert slave.positions == {}
    assert engine.stats["failed"] == 1
    assert sum(1 for c in slave.calls if c[0] == "open") == 2


def test_open_retry_succeeds_after_transient_failure(tmp_path):
    engine, master, slave = make(tmp_path, max_retries=3)
    slave.fail_opens = 1
    master.open(1, "EURUSD", "BUY", 0.10)
    engine.poll_once()
    assert len(slave.positions) == 1 and engine.stats["copied"] == 1


def test_no_duplicate_open_for_already_linked_ticket(tmp_path):
    engine, master, slave = make(tmp_path)
    master.open(1, "EURUSD", "BUY", 0.10)
    engine.poll_once()
    engine.handle_open(master.get_positions()[0])   # simulate a repeated open event
    assert len(slave.positions) == 1


def test_modify_propagates_sl_tp(tmp_path):
    engine, master, slave = make(tmp_path)
    master.open(1, "EURUSD", "BUY", 0.10)
    engine.poll_once()
    master.modify(1, sl=1.05, tp=1.15)
    engine.poll_once()
    p = next(iter(slave.positions.values()))
    assert p["sl"] == 1.05 and p["tp"] == 1.15


def test_reverse_trades_flips_side_and_swaps_sl_tp(tmp_path):
    engine, master, slave = make(tmp_path, reverse_trades=True)
    master.open(1, "EURUSD", "BUY", 0.10, sl=1.09, tp=1.12)
    engine.poll_once()
    p = next(iter(slave.positions.values()))
    assert p["side"] == "SELL" and p["sl"] == 1.12 and p["tp"] == 1.09


def test_magic_whitelist_filters(tmp_path):
    engine, master, slave = make(tmp_path, magic_number_whitelist=[777])
    master.open(1, "EURUSD", "BUY", 0.10, magic=1)
    master.open(2, "EURUSD", "BUY", 0.10, magic=777)
    engine.poll_once()
    assert len(slave.positions) == 1


def test_dry_run_touches_nothing(tmp_path):
    master = FakeMaster()
    slave = FakeAdapter("IF")
    slave.connect()
    engine = CopierEngine(master, state_path=tmp_path / "p.json", dry_run=True, heartbeat_s=9999)
    engine.add_slave(SlaveConfig.from_dict({"name": "IF", "platform": "fake"}, slave))
    engine.reconcile_on_start()
    master.open(1, "EURUSD", "BUY", 0.10)
    engine.poll_once()
    master.close(1)
    engine.poll_once()
    assert slave.positions == {} and not any(c[0] in ("open", "close") for c in slave.calls)


def test_heartbeat_reconnects_when_session_lost(tmp_path):
    engine, master, slave = make(tmp_path)
    slave._connected = False
    engine.heartbeat()
    assert slave.is_connected()

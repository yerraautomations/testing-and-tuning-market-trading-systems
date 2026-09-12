"""
Copier engine: mirrors master (MT5) positions to slave adapters, safely.

Safety properties (each one addresses a failure of the old copier):
  * every slave open is VERIFIED (the adapter only returns once the position
    is visible) and its id is stored on disk (state/positions.json);
  * closes are verified and retried; a close that keeps failing raises an
    alert instead of being silently "attempted";
  * on startup the engine reconciles: master positions that closed while the
    copier was down are closed on the slaves; slave positions the engine
    does not know about are reported (and optionally closed);
  * partial closes on the master close the proportional part on the slave;
  * volumes are normalized to the slave symbol's min / step / max;
  * a heartbeat polls the slave and reconnects when the session drops;
  * dry_run logs every action without touching the slave.
"""

import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from .position import Position, PositionManager
from .symbol_mapper import SymbolMapper
from .state import StateStore

logger = logging.getLogger("trade_copier")


@dataclass
class SlaveConfig:
    name: str
    platform: str
    adapter: Any
    enabled: bool = True
    symbol_mapper: Optional[SymbolMapper] = None
    symbol_whitelist: List[str] = field(default_factory=list)
    symbol_blacklist: List[str] = field(default_factory=list)
    volume_mode: str = "multiplier"        # multiplier | fixed | ratio
    lot_multiplier: float = 1.0
    fixed_lot: float = 0.01
    master_balance_for_ratio: float = 0.0
    copy_tp: bool = True
    copy_sl: bool = True
    reverse_trades: bool = False
    magic_whitelist: List[int] = field(default_factory=list)
    magic_blacklist: List[int] = field(default_factory=list)
    protection_enabled: bool = False
    min_balance: float = 0.0
    max_balance: float = 0.0
    max_daily_loss: float = 0.0
    close_orphans_on_start: bool = True      # master closed while copier was down -> close slave
    close_unknown_slave_positions: bool = False   # slave positions we never opened: close (True) or only warn
    copy_existing_on_start: bool = False     # master positions already open at start
    max_retries: int = 3

    @classmethod
    def from_dict(cls, d: dict, adapter) -> "SlaveConfig":
        prot = d.get("protection") or {}
        return cls(
            name=d.get("name", "slave"), platform=d.get("platform", ""), adapter=adapter,
            enabled=d.get("enabled", True),
            symbol_mapper=SymbolMapper(d.get("symbol_mapping") or {}, d.get("symbol_suffix") or "",
                                       d.get("auto_symbol_mapping", True)),
            symbol_whitelist=d.get("symbol_whitelist") or [],
            symbol_blacklist=d.get("symbol_blacklist") or [],
            volume_mode=d.get("volume_mode", "multiplier"),
            lot_multiplier=float(d.get("lot_multiplier", 1.0)),
            fixed_lot=float(d.get("fixed_lot", 0.01)),
            master_balance_for_ratio=float(d.get("master_balance_for_ratio", 0.0)),
            copy_tp=d.get("copy_tp", True), copy_sl=d.get("copy_sl", True),
            reverse_trades=d.get("reverse_trades", False),
            magic_whitelist=d.get("magic_number_whitelist") or [],
            magic_blacklist=d.get("magic_number_blacklist") or [],
            protection_enabled=prot.get("enabled", False),
            min_balance=float(prot.get("min_balance", 0) or 0),
            max_balance=float(prot.get("max_balance", 0) or 0),
            max_daily_loss=float(prot.get("max_daily_loss", 0) or 0),
            close_orphans_on_start=d.get("close_orphans_on_start", True),
            close_unknown_slave_positions=d.get("close_unknown_slave_positions", False),
            copy_existing_on_start=d.get("copy_existing_on_start", False),
            max_retries=int(d.get("max_retries", 3)),
        )


class CopierEngine:
    def __init__(self, master_monitor, state_path="state/positions.json", poll_interval_ms=250,
                 heartbeat_s=30, dry_run=False, notifier=None):
        self.master = master_monitor
        self.poll_interval = poll_interval_ms / 1000.0
        self.heartbeat_s = heartbeat_s
        self.dry_run = dry_run
        self.notifier = notifier
        self.state = StateStore(state_path)
        self.positions = PositionManager()
        self.slaves: Dict[str, SlaveConfig] = {}
        self._running = False
        self._last_heartbeat = 0.0
        self.stats = {"copied": 0, "closed": 0, "reduced": 0, "modified": 0, "failed": 0,
                      "skipped": 0, "start_time": None}

    # -- setup ----------------------------------------------------------------
    def add_slave(self, cfg: SlaveConfig):
        self.slaves[cfg.name] = cfg
        logger.info(f"Slave added: {cfg.name} ({cfg.platform}){' [DRY RUN]' if self.dry_run else ''}")

    def alert(self, msg, level="error"):
        logger.log(logging.ERROR if level == "error" else logging.WARNING, msg)
        if self.notifier:
            try:
                self.notifier.send(msg, level)
            except Exception:
                pass

    # -- main loop ------------------------------------------------------------
    def start(self):
        self._running = True
        self.stats["start_time"] = datetime.now()
        self.reconcile_on_start()
        logger.info("Copier running.")
        while self._running:
            try:
                self.poll_once()
                if time.time() - self._last_heartbeat > self.heartbeat_s:
                    self.heartbeat()
                time.sleep(self.poll_interval)
            except KeyboardInterrupt:
                self.stop()
            except Exception as e:
                logger.error(f"Error in copier loop: {e}", exc_info=True)
                time.sleep(1)

    def stop(self):
        self._running = False
        logger.info(f"Stopped. Stats: {self.stats}")

    def poll_once(self):
        current = self.master.get_positions()
        changes = self.positions.update_positions(current)
        for pos in changes["opened"]:
            self.handle_open(pos)
        for pos in changes["closed"]:
            self.handle_close(pos)
        for pos, closed_volume in changes["reduced"]:
            self.handle_reduce(pos, closed_volume)
        for pos in changes["modified"]:
            self.handle_modify(pos)

    # -- reconciliation -------------------------------------------------------
    def reconcile_on_start(self):
        """Bring disk state, master and slaves back into agreement."""
        master_positions = self.master.get_positions()
        master_by_ticket = {p.ticket: p for p in master_positions}
        self.positions.seed(master_positions)
        logger.info(f"Startup: {len(master_positions)} master positions, "
                    f"{len(self.state.links)} links on disk")

        # 1. master closed while we were down -> close the slave copies
        for ticket in sorted(self.state.tracked_tickets()):
            if int(ticket) not in master_by_ticket:
                link = self.state.links[ticket]
                logger.warning(f"Master ticket {ticket} ({link.get('symbol')}) closed while copier was down")
                for name, s in list(link["slaves"].items()):
                    cfg = self.slaves.get(name)
                    if cfg and cfg.close_orphans_on_start:
                        self._close_on_slave(cfg, ticket, s["id"], None, reason="orphan on start")
                    else:
                        logger.warning(f"  leaving {name} position {s['id']} open (close_orphans_on_start=false)")
                self.state.forget(ticket)

        # 2. restore slave links for still-open master positions
        for ticket, link in self.state.links.items():
            pos = master_by_ticket.get(int(ticket))
            if pos:
                for name, s in link["slaves"].items():
                    self.positions.add_slave_ticket(pos.ticket, name, s["id"])

        # 3. master positions we never linked (opened while down, or before first run)
        for pos in master_positions:
            if str(pos.ticket) not in self.state.links:
                if any(c.copy_existing_on_start for c in self.slaves.values()):
                    logger.info(f"Copying pre-existing master position {pos.ticket} {pos.symbol}")
                    self.handle_open(pos)
                else:
                    logger.info(f"Not copying pre-existing master position {pos.ticket} {pos.symbol} "
                                f"(copy_existing_on_start=false)")

        # 4. slave positions the engine does not know about
        self.check_unknown_slave_positions()

    def check_unknown_slave_positions(self):
        for name, cfg in self.slaves.items():
            if not cfg.enabled:
                continue
            try:
                slave_positions = cfg.adapter.get_positions()
            except Exception as e:
                self.alert(f"{name}: cannot list positions ({e})")
                continue
            known = self.state.linked_slave_ids(name)
            for sp in slave_positions:
                if str(sp["id"]) not in known:
                    msg = (f"{name}: unknown position {sp['id']} {sp['symbol']} {sp['side']} "
                           f"{sp['volume']} (not opened by this copier)")
                    if cfg.close_unknown_slave_positions and not self.dry_run:
                        self.alert(msg + " -> closing", "warning")
                        cfg.adapter.close_position(sp["id"])
                    else:
                        self.alert(msg + " -> left open (close_unknown_slave_positions=false)", "warning")

    def heartbeat(self):
        self._last_heartbeat = time.time()
        for name, cfg in self.slaves.items():
            if not cfg.enabled:
                continue
            try:
                ok = cfg.adapter.is_connected() and cfg.adapter.get_positions() is not None
            except Exception as e:
                logger.warning(f"{name}: heartbeat failed ({e})")
                ok = False
            if not ok:
                self.alert(f"{name}: session lost, reconnecting", "warning")
                try:
                    cfg.adapter.disconnect()
                    if cfg.adapter.connect():
                        logger.info(f"{name}: reconnected")
                    else:
                        self.alert(f"{name}: reconnect FAILED -- copying to {name} is suspended")
                except Exception as e:
                    self.alert(f"{name}: reconnect error {e}")

    # -- handlers -------------------------------------------------------------
    def handle_open(self, pos: Position):
        self.state.track_master(pos.ticket, pos.symbol, pos.direction, pos.volume)
        for name, cfg in self.slaves.items():
            if not cfg.enabled:
                continue
            try:
                self._copy_to_slave(pos, cfg)
            except Exception as e:
                self.stats["failed"] += 1
                self.alert(f"{name}: copy of {pos.symbol} ticket {pos.ticket} failed: {e}")

    def handle_close(self, pos: Position):
        for name, cfg in self.slaves.items():
            if not cfg.enabled:
                continue
            link = self.state.slave_of(pos.ticket, name)
            if not link:
                logger.info(f"{name}: no slave position linked to ticket {pos.ticket}; nothing to close")
                continue
            self._close_on_slave(cfg, pos.ticket, link["id"], None, reason="master closed")
        self.state.forget(pos.ticket)

    def handle_reduce(self, pos: Position, closed_volume: float):
        for name, cfg in self.slaves.items():
            if not cfg.enabled:
                continue
            link = self.state.slave_of(pos.ticket, name)
            if not link:
                continue
            master_before = pos.volume + closed_volume
            part = link["volume"] * closed_volume / master_before
            part = self._normalize_volume(cfg, self._slave_symbol(cfg, pos.symbol), part, allow_zero=True)
            if part <= 0:
                logger.info(f"{name}: partial close too small to mirror ({closed_volume} master)")
                continue
            if part >= link["volume"] - 1e-9:
                self._close_on_slave(cfg, pos.ticket, link["id"], None, reason="reduce covers whole slave position")
            elif self._close_on_slave(cfg, pos.ticket, link["id"], part, reason="partial close"):
                self.state.set_slave_volume(pos.ticket, name, round(link["volume"] - part, 8))
                self.stats["reduced"] += 1
        self.state.track_master(pos.ticket, pos.symbol, pos.direction, pos.volume)

    def handle_modify(self, pos: Position):
        for name, cfg in self.slaves.items():
            if not cfg.enabled or not (cfg.copy_sl or cfg.copy_tp):
                continue
            link = self.state.slave_of(pos.ticket, name)
            if not link:
                continue
            sl = pos.stop_loss if cfg.copy_sl else None
            tp = pos.take_profit if cfg.copy_tp else None
            if self.dry_run:
                logger.info(f"[DRY RUN] {name}: modify {link['id']} SL={sl} TP={tp}")
                continue
            if cfg.adapter.modify_position(link["id"], sl, tp):
                self.stats["modified"] += 1
            else:
                logger.warning(f"{name}: modify of {link['id']} not applied")

    # -- helpers --------------------------------------------------------------
    def _slave_symbol(self, cfg, master_symbol):
        return cfg.symbol_mapper.map_symbol(master_symbol) if cfg.symbol_mapper else master_symbol

    def _allowed(self, pos, cfg):
        if cfg.magic_whitelist and pos.magic_number not in cfg.magic_whitelist:
            return False, "magic not in whitelist"
        if cfg.magic_blacklist and pos.magic_number in cfg.magic_blacklist:
            return False, "magic blacklisted"
        if cfg.symbol_mapper and not cfg.symbol_mapper.is_symbol_allowed(
                pos.symbol, cfg.symbol_whitelist, cfg.symbol_blacklist):
            return False, "symbol filtered"
        return True, ""

    def _protection_ok(self, cfg):
        if not cfg.protection_enabled:
            return True
        info = cfg.adapter.get_account_info()
        if not info:
            return True
        bal = float(info.get("balance", 0) or 0)
        if cfg.min_balance > 0 and bal < cfg.min_balance:
            self.alert(f"{cfg.name}: balance {bal} below {cfg.min_balance} -- trading suspended", "warning")
            return False
        if cfg.max_balance > 0 and bal > cfg.max_balance:
            return False
        return True

    def _raw_volume(self, master_volume, cfg):
        if cfg.volume_mode == "fixed":
            return cfg.fixed_lot
        if cfg.volume_mode == "ratio" and cfg.master_balance_for_ratio > 0:
            info = cfg.adapter.get_account_info() or {}
            bal = float(info.get("balance", 0) or 0)
            return master_volume * (bal / cfg.master_balance_for_ratio) if bal > 0 else master_volume
        return master_volume * cfg.lot_multiplier

    def _normalize_volume(self, cfg, symbol, volume, allow_zero=False):
        spec = None
        try:
            spec = cfg.adapter.get_symbol_spec(symbol)
        except Exception:
            pass
        spec = spec or {"min_volume": 0.01, "volume_step": 0.01, "max_volume": 1000.0}
        step = spec.get("volume_step") or 0.01
        vmin = spec.get("min_volume") or step
        vmax = spec.get("max_volume") or 1e9
        v = math.floor(volume / step + 1e-9) * step
        v = round(v, 8)
        if v < vmin:
            return 0.0 if allow_zero else (vmin if volume >= vmin * 0.5 else 0.0)
        return min(v, vmax)

    def _copy_to_slave(self, pos: Position, cfg: SlaveConfig):
        ok, why = self._allowed(pos, cfg)
        if not ok:
            logger.info(f"{cfg.name}: skipping {pos.symbol} ticket {pos.ticket} ({why})")
            self.stats["skipped"] += 1
            return
        if not self._protection_ok(cfg):
            self.stats["skipped"] += 1
            return
        symbol = self._slave_symbol(cfg, pos.symbol)
        volume = self._normalize_volume(cfg, symbol, self._raw_volume(pos.volume, cfg))
        if volume <= 0:
            self.alert(f"{cfg.name}: {pos.symbol} volume {pos.volume} x settings is below the minimum "
                       f"lot -- NOT copied", "warning")
            self.stats["skipped"] += 1
            return
        side = pos.direction
        if cfg.reverse_trades:
            side = "SELL" if side == "BUY" else "BUY"
        sl = pos.stop_loss if cfg.copy_sl else 0.0
        tp = pos.take_profit if cfg.copy_tp else 0.0
        if cfg.reverse_trades and (sl or tp):
            sl, tp = tp, sl

        if self.dry_run:
            logger.info(f"[DRY RUN] {cfg.name}: open {symbol} {side} {volume} SL={sl} TP={tp} (master {pos.ticket})")
            return

        # Never open twice for the same master ticket (e.g. after a crash mid-open)
        if self.state.slave_of(pos.ticket, cfg.name):
            logger.warning(f"{cfg.name}: ticket {pos.ticket} already linked; not opening again")
            return

        for attempt in range(1, cfg.max_retries + 1):
            result = cfg.adapter.open_position(symbol, side, volume, sl, tp, comment=f"copy:{pos.ticket}")
            if result and result.get("id") is not None:
                self.state.link(pos.ticket, cfg.name, result["id"], float(result.get("volume", volume)))
                self.positions.add_slave_ticket(pos.ticket, cfg.name, result["id"])
                self.stats["copied"] += 1
                logger.info(f"{cfg.name}: opened {symbol} {side} {volume} -> id {result['id']} (master {pos.ticket})")
                return
            logger.warning(f"{cfg.name}: open attempt {attempt}/{cfg.max_retries} failed for {symbol}")
            time.sleep(0.5 * attempt)
        self.stats["failed"] += 1
        self.alert(f"{cfg.name}: FAILED to open {symbol} {side} {volume} for master ticket {pos.ticket} "
                   f"after {cfg.max_retries} attempts")

    def _close_on_slave(self, cfg, ticket, slave_id, volume, reason=""):
        if self.dry_run:
            logger.info(f"[DRY RUN] {cfg.name}: close {slave_id} volume={volume or 'all'} ({reason})")
            return True
        for attempt in range(1, cfg.max_retries + 1):
            try:
                if cfg.adapter.close_position(slave_id, volume):
                    self.stats["closed"] += 1
                    logger.info(f"{cfg.name}: closed {slave_id} volume={volume or 'all'} ({reason})")
                    return True
            except Exception as e:
                logger.warning(f"{cfg.name}: close error {e}")
            # maybe it is already gone
            try:
                if str(slave_id) not in {str(p["id"]) for p in cfg.adapter.get_positions()}:
                    logger.info(f"{cfg.name}: {slave_id} already closed")
                    return True
            except Exception:
                pass
            time.sleep(0.5 * attempt)
        self.stats["failed"] += 1
        self.alert(f"{cfg.name}: FAILED to close position {slave_id} (master {ticket}) after "
                   f"{cfg.max_retries} attempts -- CLOSE IT MANUALLY")
        return False

    def status(self):
        return {"running": self._running, "dry_run": self.dry_run, "links": len(self.state.links),
                "stats": self.stats,
                "slaves": {n: {"enabled": c.enabled, "connected": c.adapter.is_connected()}
                           for n, c in self.slaves.items()}}

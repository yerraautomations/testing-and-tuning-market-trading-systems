"""
Persistent copier state: which slave position belongs to which master ticket.

Written to state/positions.json after every change so a restart (or crash)
does not lose the links. Without this, a master close after a restart could
not be matched to its slave position -- the failure mode of the old copier.
"""

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional


class StateStore:
    def __init__(self, path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self.links: Dict[str, Dict] = {}   # master_ticket (str) -> {symbol, side, volume, slaves: {name: {id, volume}}}
        self.load()

    def load(self):
        if self.path.exists():
            try:
                self.links = json.loads(self.path.read_text(encoding="utf-8")).get("links", {})
            except Exception:
                self.links = {}
        return self.links

    def save(self):
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"saved_at": datetime.now().isoformat(), "links": self.links},
                                      indent=2), encoding="utf-8")
            tmp.replace(self.path)

    def track_master(self, ticket, symbol, side, volume):
        key = str(ticket)
        self.links.setdefault(key, {"symbol": symbol, "side": side, "volume": volume, "slaves": {}})
        self.links[key]["volume"] = volume
        self.save()

    def link(self, ticket, slave_name, slave_id, slave_volume):
        key = str(ticket)
        self.links.setdefault(key, {"symbol": "", "side": "", "volume": 0.0, "slaves": {}})
        self.links[key]["slaves"][slave_name] = {"id": str(slave_id), "volume": slave_volume}
        self.save()

    def slave_of(self, ticket, slave_name) -> Optional[Dict]:
        return self.links.get(str(ticket), {}).get("slaves", {}).get(slave_name)

    def set_slave_volume(self, ticket, slave_name, volume):
        s = self.slave_of(ticket, slave_name)
        if s is not None:
            s["volume"] = volume
            self.save()

    def unlink_slave(self, ticket, slave_name):
        key = str(ticket)
        if key in self.links:
            self.links[key]["slaves"].pop(slave_name, None)
            if not self.links[key]["slaves"]:
                self.links.pop(key)
            self.save()

    def forget(self, ticket):
        if self.links.pop(str(ticket), None) is not None:
            self.save()

    def tracked_tickets(self):
        return set(self.links)

    def linked_slave_ids(self, slave_name):
        return {v["slaves"][slave_name]["id"]: k for k, v in self.links.items() if slave_name in v["slaves"]}

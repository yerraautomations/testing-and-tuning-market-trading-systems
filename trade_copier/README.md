# Trade Copier 2.0

Copies positions from a MetaTrader 5 account (master) to prop-firm accounts
on other platforms (slaves), without a paid copier service.

Version 2.0 is a rebuild of the execution side. The 1.x browser-clicking
adapter could not tell whether an order had really filled, guessed which
position to close, and forgot its links on restart. Those are exactly the
failures that blow a funded account, so 2.0 is built around three rules:

1. **Every slave open is verified** (the adapter returns the platform's
   position id only once the position is visible) and **stored on disk**.
2. **Every close is verified and retried**; if it still fails you get an
   alert that says "close it manually" instead of a silent "attempted".
3. **Restarts reconcile**: positions the master closed while the copier was
   down are closed on the slaves; slave positions the copier never opened
   are reported (and closed only if you say so).

Plus: partial closes are mirrored proportionally, volumes are normalized to
the slave's min/step/max lot rules, a heartbeat re-logs-in when a session
drops, and `--dry-run` shows what would happen without trading.

## Layout

```
trade_copier/
  main.py                     entry point (--test, --dry-run, --headed)
  config/settings.yaml        template (committed); real values go in
  config/settings.local.yaml  (git-ignored) or env vars TC_SLAVE_PASSWORD etc.
  config/servers.yaml         MatchTrader prop-firm URLs
  core/copier_engine.py       the engine (verified opens/closes, reconciliation)
  core/state.py               state/positions.json: master ticket -> slave id
  core/position.py            master position model + change detection
  core/adapter_base.py        the contract every slave adapter must fulfil
  core/fake_adapter.py        simulated platform for tests and dry runs
  forex/mt5_monitor.py        master: MetaTrader5 position polling
  forex/matchtrader_web.py    slave: MatchTrader via the page's own web calls (built from a recording)
  forex/matchtrader_adapter.py  slave: REST adapter for firms that allow API access (unverified endpoints)
  forex/matchtrader_browser.py  LEGACY 1.x click-based adapter -- do not use
  tools/record_session.py     records the platform's network calls while you trade by hand
  tests/test_engine.py        engine tests against the fake platform (pytest)
```

## Setup

```
pip install -r requirements.txt
python -m playwright install chromium
```
Put your real values in `config/settings.local.yaml` (copy the slave block
from `settings.yaml`). Never commit it. If a password was ever pasted into a
chat or a cloud session, change it.

## Step 1: record how the platform trades (once per prop firm)

The MatchTrader website is a web app talking to a backend. The copier must
speak to that backend the same way the page does. Record it:

```
python tools/record_session.py --url https://platform.instantfunding.com
```
A Chrome window opens. Log in yourself (Cloudflare, 2FA), close popups, wait
10 s, open ONE minimum-size position, wait 10 s, close it, wait 10 s, then
press Enter in the console. The recording (secrets masked) lands in
`recordings/`. The browser profile in `browser_profile/` keeps you logged in
for later runs. Use a demo account if the firm offers one.

From the recording the web adapter (`forex/matchtrader_web.py`) is written:
which calls list positions, open, close, modify, and where the session token
lives. Until it exists, `main.py` refuses to start a browser slave.

## Step 2: test, dry-run, then go live on a demo first

```
python main.py --test          # MT5 + slave login + lists open positions
python main.py --dry-run       # engine runs, logs every action, trades nothing
python main.py                 # real copying
```
Run `pytest tests -q` after any engine change (17 tests).

## Safety settings (per slave, in settings.yaml)

| Key | Default | Meaning |
|---|---|---|
| `close_orphans_on_start` | true | master closed while copier was down -> close the slave copy |
| `close_unknown_slave_positions` | false | slave positions not opened by the copier: warn (false) or close (true) |
| `copy_existing_on_start` | false | copy master positions that are already open at start |
| `max_retries` | 3 | open/close attempts before alerting |
| `protection.min_balance` | 0 | suspend copying below this slave balance |

## Known limits

* One master (MT5) for now. Futures adapters are not written.
* Prop firms differ on whether copiers and automation are allowed; check the
  rules of the firm you use. The REST adapter's endpoints are unverified.
* Latency is the poll interval plus the platform's own response time; expect
  0.3 to 2 s. Not suitable for scalping strategies.

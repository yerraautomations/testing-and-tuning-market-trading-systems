"""
Record the MatchTrader web platform's own network calls while YOU trade by hand.

Why: the copier must place and close trades exactly the way the website does,
with real confirmations and position IDs. The reliable way to learn those
calls is to watch the page make them. This tool opens a real Chrome window,
you log in and open/close ONE tiny trade (use a demo account if the firm
offers one), and every request, response and websocket frame is written to
a JSON-lines file with secrets masked.

    python tools/record_session.py --url https://platform.instantfunding.com

Steps when the window opens:
  1. Log in (solve any Cloudflare / 2FA yourself). Dismiss popups.
  2. Wait 10 seconds so the positions/account calls are captured.
  3. Open ONE position at minimum size, wait 10 seconds.
  4. Close it, wait 10 seconds.
  5. Come back to this console and press Enter.

The browser profile is kept in trade_copier/browser_profile/<host>/ so the
login survives restarts; the copier itself reuses that profile.
"""

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent.parent
SECRET_HEADERS = {"authorization", "cookie", "set-cookie", "x-auth-token", "x-csrf-token",
                  "x-xsrf-token", "authentication"}
SECRET_KEYS = re.compile(r"(password|passwd|token|secret|cookie|session|jwt|authorization)", re.I)
MAX_BODY = 200_000


def mask_headers(headers):
    out = {}
    for k, v in headers.items():
        if k.lower() in SECRET_HEADERS:
            out[k] = f"<masked len={len(v)} starts={v[:6]}>"
        else:
            out[k] = v
    return out


def mask_json(obj):
    if isinstance(obj, dict):
        return {k: ("<masked>" if SECRET_KEYS.search(str(k)) else mask_json(v)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [mask_json(v) for v in obj]
    return obj


def mask_body(text):
    if not text:
        return text
    try:
        return mask_json(json.loads(text))
    except Exception:
        return SECRET_KEYS.sub(lambda m: m.group(0), text[:MAX_BODY])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True, help="platform URL, e.g. https://platform.instantfunding.com")
    ap.add_argument("--out", help="output .jsonl (default recordings/<host>_<time>.jsonl)")
    ap.add_argument("--all", action="store_true", help="also record static assets (images, css, js)")
    a = ap.parse_args()

    host = urlparse(a.url).netloc
    profile = HERE / "browser_profile" / host
    profile.mkdir(parents=True, exist_ok=True)
    out = Path(a.out) if a.out else HERE / "recordings" / f"{host}_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    f = out.open("w", encoding="utf-8")
    n = {"req": 0, "res": 0, "ws": 0}

    def write(kind, data):
        f.write(json.dumps({"t": time.time(), "kind": kind, **data}, default=str) + "\n")
        f.flush()

    def interesting(req):
        if a.all:
            return True
        return req.resource_type in ("xhr", "fetch", "websocket", "document", "other")

    with sync_playwright() as p:
        ctx = p.chromium.launch_persistent_context(
            str(profile), headless=False, viewport={"width": 1600, "height": 950},
            args=["--disable-blink-features=AutomationControlled"])
        page = ctx.pages[0] if ctx.pages else ctx.new_page()

        def on_request(req):
            if not interesting(req):
                return
            n["req"] += 1
            write("request", {"id": id(req), "method": req.method, "url": req.url,
                              "type": req.resource_type, "headers": mask_headers(req.headers),
                              "body": mask_body(req.post_data)})

        def on_response(res):
            req = res.request
            if not interesting(req):
                return
            n["res"] += 1
            body = None
            ctype = res.headers.get("content-type", "")
            if "json" in ctype or "text" in ctype:
                try:
                    txt = res.text()
                    body = mask_body(txt[:MAX_BODY])
                except Exception as e:
                    body = f"<unreadable: {e}>"
            write("response", {"id": id(req), "status": res.status, "url": res.url,
                               "method": req.method, "headers": mask_headers(res.headers), "body": body})

        def on_websocket(ws):
            write("websocket_open", {"url": ws.url})
            ws.on("framesent", lambda payload: (n.__setitem__("ws", n["ws"] + 1),
                                                write("ws_sent", {"url": ws.url, "payload": mask_body(payload)})))
            ws.on("framereceived", lambda payload: (n.__setitem__("ws", n["ws"] + 1),
                                                    write("ws_recv", {"url": ws.url, "payload": mask_body(payload)})))
            ws.on("close", lambda: write("websocket_close", {"url": ws.url}))

        ctx.on("request", on_request)
        ctx.on("response", on_response)
        page.on("websocket", on_websocket)
        page.on("framenavigated", lambda fr: write("navigated", {"url": fr.url}) if fr == page.main_frame else None)

        print(f"Recording to {out}")
        print("Log in, open ONE tiny trade, close it, then press Enter here.")
        page.goto(a.url, wait_until="domcontentloaded", timeout=120_000)
        try:
            input()
        except EOFError:
            time.sleep(600)

        # Where does the app keep its session? Names only, values masked.
        try:
            storage = page.evaluate("""() => ({
                localStorage: Object.fromEntries(Object.keys(localStorage).map(k => [k, String(localStorage.getItem(k)).slice(0, 12) + '...'])),
                sessionStorage: Object.fromEntries(Object.keys(sessionStorage).map(k => [k, String(sessionStorage.getItem(k)).slice(0, 12) + '...'])),
                url: location.href })""")
            write("storage", storage)
        except Exception as e:
            write("storage", {"error": str(e)})
        cookies = ctx.cookies()
        write("cookies", {"names": [{"name": c["name"], "domain": c["domain"], "httpOnly": c.get("httpOnly")} for c in cookies]})
        try:
            page.screenshot(path=str(out.with_suffix(".png")), full_page=False)
        except Exception:
            pass
        ctx.close()

    f.close()
    print(f"Done. {n['req']} requests, {n['res']} responses, {n['ws']} websocket frames -> {out}")


if __name__ == "__main__":
    main()

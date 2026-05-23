"""Diagnose whether Betfair API access from this host is blocked at the
network/IP level, or whether the original 403 was just auth-related.

Three probes:
  1. Public website fetch (any IP can reach this normally)
  2. Identity SSO login endpoint with deliberately bad credentials
     - If reachable: returns INVALID_USERNAME_OR_PASSWORD (we know auth path works)
     - If blocked: connection refused / 403 / Cloudflare challenge
  3. (Optional) JSON-RPC endpoint with no session token
     - If reachable: returns NO_SESSION error
     - If blocked: same as above

Run on Hetzner (or wherever you suspect blocking). Compare to running it
from your laptop -- if laptop reaches all three but Hetzner doesn't, the IP
is blocked. If both reach all three, the original 403 was about credentials.
"""
from __future__ import annotations

import json
import socket
import sys
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError


def probe(label: str, url: str, *, method: str = "GET", body: bytes | None = None,
          headers: dict | None = None, timeout: int = 10) -> None:
    print(f"\n=== {label} ===")
    print(f"  {method} {url}")
    try:
        req = Request(url, data=body, method=method, headers=headers or {})
        with urlopen(req, timeout=timeout) as r:
            status = r.status
            content_type = r.headers.get("Content-Type", "")
            data = r.read(2000)
            print(f"  status: {status}  content-type: {content_type}")
            if "json" in content_type:
                try:
                    print(f"  body:   {json.loads(data)}")
                except Exception:
                    print(f"  body[:200]: {data[:200]!r}")
            else:
                # Look for Cloudflare / block markers in HTML
                low = data.lower()
                markers = [m for m in (b"cloudflare", b"access denied", b"forbidden",
                                       b"captcha", b"ray id") if m in low]
                if markers:
                    print(f"  *** block markers found: {markers}")
                else:
                    print(f"  HTML ok, body[:200]: {data[:200]!r}")
    except HTTPError as e:
        print(f"  HTTPError: {e.code} {e.reason}")
        try:
            err_body = e.read(1000)
            print(f"  body[:200]: {err_body[:200]!r}")
        except Exception:
            pass
    except URLError as e:
        print(f"  URLError: {e.reason}")
    except socket.timeout:
        print(f"  TIMEOUT after {timeout}s")
    except Exception as e:
        print(f"  OTHER: {type(e).__name__}: {e}")


def main() -> int:
    print("Betfair reachability probe -- running from this host")
    print(f"  python: {sys.version.split()[0]}")
    try:
        import urllib.request
        ip = urlopen("https://api.ipify.org", timeout=5).read().decode().strip()
        print(f"  outbound IP: {ip}")
    except Exception as e:
        print(f"  outbound IP: unknown ({e})")

    probe("1. Public homepage",
          "https://www.betfair.com/exchange/plus/cricket")

    probe("2. Identity SSO login (deliberate bad creds)",
          "https://identitysso.betfair.com/api/login",
          method="POST",
          body=b"username=invalid_test_user&password=invalid_test_pass",
          headers={"Content-Type": "application/x-www-form-urlencoded",
                   "X-Application": "test", "Accept": "application/json"})

    probe("3. Exchange JSON-RPC (no session token)",
          "https://api.betfair.com/exchange/betting/json-rpc/v1",
          method="POST",
          body=b'{"jsonrpc":"2.0","method":"SportsAPING/v1.0/listEventTypes","params":{"filter":{}},"id":1}',
          headers={"Content-Type": "application/json",
                   "X-Application": "test", "Accept": "application/json"})

    print("\n--- Interpretation ---")
    print("  All three reachable (200 / structured-error JSON): network OK, original 403 was credentials")
    print("  #1 OK, #2/#3 blocked (Cloudflare/403/timeout): IP allowed for web but blocked for API endpoints")
    print("  All three blocked: data-center IP block -- need Cloudflare Tunnel or residential proxy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

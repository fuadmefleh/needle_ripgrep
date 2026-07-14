"""Fast NGT client: talks to a running server.py over a Unix socket.

Requires `server.py` to already be running (it does the one-time model
load + JIT warmup). This client is cheap to invoke repeatedly -- each call
is just a socket round-trip plus one `rg` subprocess.

Usage:
    ./needle/.venv/bin/python client.py "find TODO comments"
    ./needle/.venv/bin/python client.py --path src/ --dry-run "find async functions starting with fetch"
"""

import argparse
import os
import socket
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cli import build_rg_command  # noqa: E402
from ipc import recv_json_line, send_json_line  # noqa: E402
from server import SOCKET_PATH  # noqa: E402


def main():
    p = argparse.ArgumentParser(description="Fast NGT client (requires server.py running)")
    p.add_argument("query", nargs="+")
    p.add_argument("--path", default=".")
    p.add_argument("--socket", default=SOCKET_PATH)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    nl_query = " ".join(args.query)

    if not os.path.exists(args.socket):
        print(f"[ngt] no server socket at {args.socket} -- start it first with:\n"
              f"      ./needle/.venv/bin/python server.py", file=sys.stderr)
        sys.exit(1)

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    t0 = time.perf_counter()
    try:
        sock.connect(args.socket)
    except ConnectionRefusedError:
        print(f"[ngt] socket at {args.socket} exists but nothing is listening "
              f"(stale from a crashed server?) -- remove it and restart server.py", file=sys.stderr)
        sys.exit(1)

    with sock:
        send_json_line(sock, {"query": nl_query})
        resp = recv_json_line(sock)
    round_trip_ms = (time.perf_counter() - t0) * 1000

    if "error" in resp:
        print(f"[ngt] server error: {resp['error']}", file=sys.stderr)
        sys.exit(1)

    params = resp["params"]
    outcome = resp["outcome"]

    if outcome == "declined":
        print(f"[ngt] not a search query -- no tool call generated for: {nl_query!r}  "
              f"(server: {resp['latency_ms']:.0f}ms)", file=sys.stderr)
        return
    if outcome == "fallback":
        print(f"[ngt] model output failed validation; falling back to literal search for: {nl_query!r}", file=sys.stderr)

    cmd = build_rg_command(params, path=args.path)
    print(f"[ngt] {' '.join(cmd)}  (server: {resp['latency_ms']:.0f}ms, round-trip: {round_trip_ms:.0f}ms)", file=sys.stderr)

    if args.dry_run:
        return
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()

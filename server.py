"""Persistent NGT server: loads the model once, then answers translation
requests over a Unix socket at ~530ms/query steady state instead of paying
the ~12.5s load+compile cost on every invocation (see benchmark.py).

The server only translates NL -> GrepParameters JSON; it does not run
ripgrep itself (that stays in the client, matching the original design:
the model parses intent, the native binary searches the disk).

Usage:
    ./needle/.venv/bin/python server.py
    # in another shell:
    ./needle/.venv/bin/python client.py "find TODO comments"
"""

import argparse
import os
import signal
import socket
import sys
import time

# JAX preallocates ~75-90% of total GPU memory on first use by default (and
# needle's own setup script pushes that to 95% via XLA_PYTHON_CLIENT_MEM_FRACTION,
# a setting meant for large training jobs). A 26M-param model doesn't need
# anywhere near that; must be set before jax is imported anywhere below.
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "needle"))

from ipc import recv_json_line, send_json_line  # noqa: E402

SOCKET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ngt.sock")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--socket", default=SOCKET_PATH)
    args = p.parse_args()

    import cli  # reuse cli.query_to_grep_params (schema validation + fallback)

    checkpoint_path = args.checkpoint or cli.default_checkpoint()
    print(f"Loading checkpoint: {checkpoint_path}")
    t0 = time.perf_counter()
    cli._load_model(checkpoint_path)  # populates cli._model_cache
    # Warmup: triggers the one-time JIT compile for the fixed encoder shape
    # (see fast_generate.py) so the first real client request is already fast.
    cli.query_to_grep_params("find TODO", checkpoint_path=checkpoint_path)
    print(f"Ready in {time.perf_counter() - t0:.1f}s (load + warmup compile)")

    if os.path.exists(args.socket):
        os.remove(args.socket)

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(args.socket)
    server.listen(5)
    print(f"Listening on {args.socket} (Ctrl+C to stop)")

    def _cleanup(*_a):
        server.close()
        if os.path.exists(args.socket):
            os.remove(args.socket)
        sys.exit(0)

    signal.signal(signal.SIGINT, _cleanup)
    signal.signal(signal.SIGTERM, _cleanup)

    try:
        while True:
            conn, _ = server.accept()
            with conn:
                try:
                    req = recv_json_line(conn)
                    query = req["query"]
                    t0 = time.perf_counter()
                    params, outcome = cli.query_to_grep_params(query, checkpoint_path=checkpoint_path)
                    latency_ms = (time.perf_counter() - t0) * 1000
                    send_json_line(conn, {
                        "params": params,
                        "outcome": outcome,
                        "latency_ms": latency_ms,
                    })
                except Exception as e:
                    send_json_line(conn, {"error": str(e)})
    finally:
        _cleanup()


if __name__ == "__main__":
    main()

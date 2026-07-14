"""Newline-delimited JSON framing over a socket, shared by server.py/client.py."""

import json


def send_json_line(sock, obj):
    sock.sendall((json.dumps(obj) + "\n").encode())


def recv_json_line(sock, bufsize=65536):
    buf = b""
    while b"\n" not in buf:
        chunk = sock.recv(bufsize)
        if not chunk:
            if buf:
                break
            raise ConnectionError("connection closed before a full line was received")
        buf += chunk
    line, _, _rest = buf.partition(b"\n")
    return json.loads(line.decode())

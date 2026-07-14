#!/usr/bin/env bash
# Symlinks bin/ngt and bin/ngt-server onto your PATH (default: ~/.local/bin).
# Safe to re-run.
set -euo pipefail

PROJECT_DIR="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="${1:-$HOME/.local/bin}"

mkdir -p "$INSTALL_DIR"
ln -sf "$PROJECT_DIR/bin/ngt" "$INSTALL_DIR/ngt"
ln -sf "$PROJECT_DIR/bin/ngt-server" "$INSTALL_DIR/ngt-server"

echo "Linked ngt and ngt-server into $INSTALL_DIR"
case ":$PATH:" in
  *":$INSTALL_DIR:"*) echo "$INSTALL_DIR is already on PATH -- you're set." ;;
  *) echo "warning: $INSTALL_DIR is not on your PATH. Add this to your shell rc:" &&
     echo "  export PATH=\"$INSTALL_DIR:\$PATH\"" ;;
esac

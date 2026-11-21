#!/usr/bin/env bash
set -euo pipefail
INSTALL_DIR="${HOME}/.local/bin"
mkdir -p "$INSTALL_DIR"
cp "$(dirname "$0")/nala-modelforge-rs" "$INSTALL_DIR/nala-modelforge-rs"
chmod 0755 "$INSTALL_DIR/nala-modelforge-rs"
echo "Installed NALA-ModelForge-RS to $INSTALL_DIR/nala-modelforge-rs"
echo "Launch studio: $INSTALL_DIR/nala-modelforge-rs --open"
echo "MCP stdio:     $INSTALL_DIR/nala-modelforge-rs --mcp-stdio"

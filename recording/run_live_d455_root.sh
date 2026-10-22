#!/usr/bin/env bash
# Forwarding script to scripts/run_live_d455.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/scripts/run_live_d455.sh" "$@"

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTROL_SCRIPT="$(realpath "${SCRIPT_DIR}/../../go2_base_controller/loco/go2_base_control.sh")"

if [[ ! -x "$CONTROL_SCRIPT" ]]; then
  echo "Go2 control script is missing or not executable: $CONTROL_SCRIPT" >&2
  exit 1
fi

cleanup() {
  "$CONTROL_SCRIPT" stop || true
}

trap cleanup EXIT INT TERM

"$CONTROL_SCRIPT" start

while true; do
  sleep 5
done

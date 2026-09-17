#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec env DATA_SET=rgbtvg_m3fd bash "$SCRIPT_DIR/../train_single.sh" "$@"

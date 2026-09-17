#!/usr/bin/env bash
set -euo pipefail

# Backward-compatible name used by the original MMVG release.
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/train_all.sh" "$@"

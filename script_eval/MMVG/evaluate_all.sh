#!/usr/bin/env bash
set -euo pipefail

# Friendly alias for the historical all_eval.sh entry point.
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/all_eval.sh" "$@"

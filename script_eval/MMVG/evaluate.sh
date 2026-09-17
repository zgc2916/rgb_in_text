#!/usr/bin/env bash
set -euo pipefail

# Friendly alias for the historical single_eval.sh entry point.
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/single_eval.sh" "$@"

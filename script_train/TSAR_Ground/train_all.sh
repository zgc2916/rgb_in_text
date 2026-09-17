#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

bash "${SCRIPT_DIR}/ref_flir/train.sh"
bash "${SCRIPT_DIR}/ref_m3fd/train.sh"
bash "${SCRIPT_DIR}/ref_mfad/train.sh"

#!/bin/bash
# Headline GPT-2 Medium E2E result using the corrected no-momentum update.
set -euo pipefail
export METHOD=imuon SEED=110
exec bash "$(dirname "$0")/run_seed_experiment.sh"

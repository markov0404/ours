#!/usr/bin/env bash
# Clona el harness JevBench (MIT) en vendor/jevbench si no existe (directorio o symlink). Uso: scripts/fetch_jevbench.sh
set -uo pipefail
cd "$(dirname "$0")/.."
[ -e vendor/jevbench ] || git clone --depth 1 https://github.com/fstandhartinger/jevbench vendor/jevbench
ls vendor/jevbench/datasets/public/*.jsonl && wc -l vendor/jevbench/datasets/public/*.jsonl

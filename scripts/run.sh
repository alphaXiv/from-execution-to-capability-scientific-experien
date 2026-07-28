#!/usr/bin/env bash
set -euo pipefail

run_start=$(date +%s)
export PIP_DISABLE_PIP_VERSION_CHECK=1
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

echo "SCICONSOLIDATE_REPRO_START=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "GIT_COMMIT=$(git rev-parse HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python3 -m pip install -q \
  "datasets>=3.6,<4" \
  "gdown>=5.2" "hf_transfer>=0.1.9" "h5py>=3.11" \
  "numpy>=1.26" "scipy>=1.13" "sympy>=1.13" "matplotlib>=3.9" \
  "peft>=0.15,<0.17"

if [[ ! -d /tmp/SciCode/.git ]]; then
  git clone --depth 1 https://github.com/scicode-bench/SciCode.git /tmp/SciCode
fi
python3 -m pip install -q -e /tmp/SciCode

mkdir -p /tmp/scicode-data
if [[ ! -f /tmp/scicode-data/test_data.h5 ]]; then
  gdown --folder "https://drive.google.com/drive/folders/1W5GZW6_bdiDAiipuFMqdUhvUaHIj6-pR" \
    -O /tmp/scicode-data
fi
test_h5=$(find /tmp/scicode-data -type f -name 'test_data.h5' -print -quit)
if [[ -z "${test_h5}" ]]; then
  echo "FATAL: official SciCode test_data.h5 was not downloaded"
  exit 2
fi

condition=$(python3 -c 'import json; print(json.load(open("experiment.json"))["condition"])')
if [[ "$condition" == "sft" ]]; then
  python3 scripts/sft_pipeline.py --config experiment.json --test-h5 "$test_h5"
else
  python3 scripts/scicode_repro.py --config experiment.json --test-h5 "$test_h5"
fi

elapsed=$(( $(date +%s) - run_start ))
echo "RUN_WALL_SECONDS=${elapsed}"
echo "SCICONSOLIDATE_REPRO_END=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

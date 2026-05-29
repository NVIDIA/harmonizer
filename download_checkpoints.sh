#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Download all checkpoints required to run Harmonizer inference and place them
# in the directories the code expects:
#   models/                                                 <- nvidia/Harmonizer (.pkl / .pt)
#   src/checkpoints/nvidia/Cosmos-Predict2-0.6B-Text2Image  <- base Cosmos DiT + tokenizer
#
# Usage:
#   ./download_checkpoints.sh                  # download the inference checkpoints
#   ./download_checkpoints.sh --with-dataset   # also download the training dataset into data/
set -euo pipefail

# Run from the repo root (the directory containing this script) so the relative
# checkpoint paths match what the inference/training code loads.
cd "$(dirname "$0")"

if ! command -v hf >/dev/null 2>&1; then
  echo "error: 'hf' (Hugging Face CLI) not found. Install with: pip install huggingface_hub[cli]" >&2
  exit 1
fi

echo "==> Downloading Harmonizer checkpoints -> models/"
hf download nvidia/Harmonizer --local-dir models

echo "==> Downloading base Cosmos-Predict2-0.6B-Text2Image -> src/checkpoints/nvidia/Cosmos-Predict2-0.6B-Text2Image/"
hf download nvidia/Cosmos-Predict2-0.6B-Text2Image \
  --local-dir src/checkpoints/nvidia/Cosmos-Predict2-0.6B-Text2Image

if [[ "${1:-}" == "--with-dataset" ]]; then
  echo "==> Downloading training dataset -> data/"
  hf download nvidia/Harmonizer-Dataset --repo-type dataset --local-dir data
fi

echo "==> Done. Inference checkpoints are in models/ and src/checkpoints/."

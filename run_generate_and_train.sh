#!/usr/bin/env bash
set -euo pipefail

python generate_dataset.py
python train_model.py

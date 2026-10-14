#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="/home/test/miniconda3/envs/trellis2/bin/python"
CKPT="$ROOT/results/multiview_1186_flow_1024_lora/ckpts/denoiser_ema0.999_step0008000.pt"
OUT="$ROOT/results_infer/side_lora_1024"
LOG="$ROOT/results_infer/side_lora_1024_infer.log"
while [[ ! -s "$CKPT" ]]; do
  sleep 30
done
mkdir -p "$OUT"
cd "$ROOT"
exec "$PY" scripts/infer_finetuned.py \
  --input_dir side \
  --output "$OUT" \
  --pipeline_type 1024 \
  --flow_1024_lora "$CKPT" \
  --lora_rank 8 --lora_alpha 8 --lora_dropout 0 \
  --no_video --max_faces 100000 >> "$LOG" 2>&1

#!/usr/bin/env bash
# ── 中文摘要 ─────────────────────────────────────────────
# 用途：守候型脚本：等 1024 LoRA 的 8000 步 EMA 权重出现后，立即对 side/ 目录做一次 1024 推理（面数上限 10 万）。
# 用法：bash scripts/training_history/watch_lora_1024_and_infer.sh（无参数；轮询 sleep 30 等待权重）。
# 输出：results_infer/side_lora_1024/（GLB）+ results_infer/side_lora_1024_infer.log。
# 边界：会 mkdir 输出目录并跑 GPU 推理（--no_video）；不训练、不改权重。
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

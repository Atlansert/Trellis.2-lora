#!/usr/bin/env bash
# ── 中文摘要 ─────────────────────────────────────────────
# 用途：Stage-2（1024 shape flow）LoRA 训练（首版配方），固定用 GPU 7。
# 用法：bash scripts/training_history/run_lora_1024.sh（无参数）；--ckpt none 从预训练权重开始。
# 输出：results/multiview_1186_flow_1024_lora/（checkpoint、log）。
# 边界：长跑 GPU 训练任务，持续写该输出目录；只读 training_data/ 与 configs/。
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="/home/test/miniconda3/envs/trellis2/bin/python"
export CUDA_VISIBLE_DEVICES=7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATA='{"train_data":{"base":"./training_data/multiview_1186","shape_latent":"./training_data/multiview_1186/shape_latents/shape_enc_next_dc_f16c32_fp16_1024","render_cond":"./training_data/multiview_1186/renders_cond"}}'
OUT="$ROOT/results/multiview_1186_flow_1024_lora"
cd "$ROOT"
exec "$PY" train.py \
  --config configs/gen/shape_flow_lora_1024_train.json \
  --output_dir "$OUT" \
  --ckpt none \
  --data_dir "$DATA" \
  --auto_retry 3

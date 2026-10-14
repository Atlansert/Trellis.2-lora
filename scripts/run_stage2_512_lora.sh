#!/usr/bin/env bash
# Stage-2b: Shape Flow 512 LoRA — the missing LR stage for 1024_cascade /
# 1536_cascade pipelines. Same recipe as the 1024 v2 run (rank8/alpha8,
# attention-only, EMA 0.9999, logitNormal(1,1), p_uncond 0.1, lr 1e-4).
# Trains on pre-encoded shape_enc_next_dc_f16c32_fp16_512 latents.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
PY="/home/test/miniconda3/envs/trellis2/bin/python"
export CUDA_VISIBLE_DEVICES=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT="$ROOT/results/multiview_1186_flow_512_lora"
DATA='{"train_data":{"base":"./training_data/multiview_1186","shape_latent":"./training_data/multiview_1186/shape_latents/shape_enc_next_dc_f16c32_fp16_512","render_cond":"./training_data/multiview_1186/renders_cond"}}'
cd "$ROOT"
exec "$PY" train.py \
  --config configs/gen/shape_flow_lora_512_train.json \
  --output_dir "$OUT" \
  --ckpt none \
  --num_gpus 1 \
  --data_dir "$DATA" \
  --auto_retry 3
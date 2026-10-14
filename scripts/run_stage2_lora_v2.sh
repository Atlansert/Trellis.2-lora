#!/usr/bin/env bash
# Stage 2 (1024 shape flow) LoRA, v2 recipe modeled on the successful Stage 1
# recipe: rank8/alpha8 on attention projections only, EMA 0.9999, logitNormal
# t-schedule, CFG p_uncond 0.1, lr 1e-4, wd 0. Keep DinoV3/1024 conditioning so
# training features match the ~inference cond_1024 of the official pipeline.
#
# Evaluate with Stage1(step030000) + this Stage2 LoRA via infer_finetuned.py.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
PY="/home/test/miniconda3/envs/trellis2/bin/python"
export CUDA_VISIBLE_DEVICES=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT="$ROOT/results/multiview_1186_flow_1024_lora_v2"
DATA='{"train_data":{"base":"./training_data/multiview_1186","shape_latent":"./training_data/multiview_1186/shape_latents/shape_enc_next_dc_f16c32_fp16_1024","render_cond":"./training_data/multiview_1186/renders_cond"}}'
cd "$ROOT"
exec "$PY" train.py \
  --config configs/gen/shape_flow_lora_1024_v2_train.json \
  --output_dir "$OUT" \
  --ckpt none \
  --num_gpus 1 \
  --data_dir "$DATA" \
  --auto_retry 3

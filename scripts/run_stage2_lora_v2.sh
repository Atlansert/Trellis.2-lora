#!/usr/bin/env bash
# ── 中文摘要 ─────────────────────────────────────────────
# 用途：Stage-2（1024 shape flow）LoRA 训练，v2 配方（r8/a8、只挂 attention、EMA 0.9999、lr 1e-4），固定 GPU 4。
# 用法：bash scripts/run_stage2_lora_v2.sh（无参数）；评估时用 infer_finetuned.py 同时挂 Stage1 与这个 Stage2 LoRA。
# 输出：results/multiview_1186_flow_1024_lora_v2/（checkpoint、log）。
# 边界：长跑 GPU 训练任务，持续写该输出目录；条件保持 DINOv3/1024 以便与官方管线口径一致。
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

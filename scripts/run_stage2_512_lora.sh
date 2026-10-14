#!/usr/bin/env bash
# ── 中文摘要 ─────────────────────────────────────────────
# 用途：Stage-2b：Shape Flow 512 LoRA 训练（1024_cascade / 1536_cascade 缺的那段 LR），固定 GPU 4。
# 用法：bash scripts/run_stage2_512_lora.sh（无参数）；训练在预编码好的 512 shape latent 上。
# 输出：results/multiview_1186_flow_512_lora/（checkpoint、log）。
# 边界：长跑 GPU 训练任务，持续写该输出目录；配方与 1024 v2 一致（r8/a8、仅 attention、EMA 0.9999、lr 1e-4）。
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
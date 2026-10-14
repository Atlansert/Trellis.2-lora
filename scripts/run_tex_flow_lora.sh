#!/usr/bin/env bash
# ── 中文摘要 ─────────────────────────────────────────────
# 用途：Texture SLat Flow (1024) LoRA 训练（r8/a8、只挂 5 个 attention 目标），以官方 imgshape2tex 1.3B 1024 为底。
# 用法：bash scripts/run_tex_flow_lora.sh [GPU 号，默认 6]；--ckpt latest 表示从已有 checkpoint 续训。
# 数据：pbr_latents + shape_latents（坐标已对齐）+ renders_cond；输出：results/multiview_1186_tex_flow_lora_1024/。
# 边界：长跑 GPU 训练任务，持续写该输出目录；只读 training_data/ 与 configs/。
# Texture SLat Flow (1024) LoRA training — rank8/alpha8, 5 attention targets,
# pretrained on official imgshape2tex 1.3B 1024, data: pbr_latents + shape_latents
# (coords-aligned) + renders_cond. Modeled on the successful stage2 shape-flow
# LoRA recipe (EMA 0.9999, logitNormal, p_uncond 0.1, lr 1e-4).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
PY="/home/test/miniconda3/envs/trellis2/bin/python"
GPU="${1:-6}"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT="$ROOT/results/multiview_1186_tex_flow_lora_1024"
DATA='{"train_data":{"base":"./training_data/multiview_1186","pbr_latent":"./training_data/multiview_1186/pbr_latents/tex_enc_next_dc_f16c32_fp16_1024","shape_latent":"./training_data/multiview_1186/shape_latents/shape_enc_next_dc_f16c32_fp16_1024","render_cond":"./training_data/multiview_1186/renders_cond"}}'
cd "$ROOT"
exec "$PY" train.py \
  --config configs/gen/tex_flow_lora_1024_train.json \
  --output_dir "$OUT" \
  --ckpt latest \
  --num_gpus 1 \
  --data_dir "$DATA" \
  --auto_retry 3
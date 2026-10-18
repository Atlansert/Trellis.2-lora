#!/usr/bin/env bash
# ── 中文摘要 ─────────────────────────────────────────────
# 用途：Stage1（SS）LoRA 训练，监督目标改为"规范帧"（latent_frame=canonical）+ DINOv3@512，50k 步。
# 用法：bash scripts/run_ss_lora_canonical_dinov3_50k.sh [GPU 号，默认 5]。
# 输出：results/multiview_1186_ss_flow_lora_64_views_canonical/。
# 边界：长跑 GPU 训练任务，持续写该输出目录；与 dinov3_512 那轮唯一差别就是监督目标口径。
# Stage1 SS LoRA —— **规范帧**目标（canonical）+ DINOv3@512 条件，50k 步。
#
# 与 ss_flow_lora_64_views_dinov3_512_50k 的唯一区别是监督目标的口径：
#   前者（latent_frame=view）: 目标 = 网格绕 z 旋转 (yaw-270°) 后的占用 latent
#   本脚本（latent_frame=canonical）: 目标 = 规范帧占用 latent
#
# 原因：实测"视角帧"与"规范帧"的占用正好相差绕 z 的 90°（旋转后 IoU=1.000），
# 而推理管线把 SS 输出直接当作规范帧使用 → 视角帧监督会让成品整体转 90°。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
PY="/home/test/miniconda3/envs/trellis2/bin/python"
GPU="${1:-5}"
OUT="$ROOT/results/multiview_1186_ss_flow_lora_64_views_canonical"
DATA='{"train_data":{"base":"./training_data/multiview_1186","ss_view_latent":"./training_data/multiview_1186/ss_latents_views_mesh_t1","render_cond":"./training_data/multiview_1186/renders_cond"}}'

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$ROOT"
exec "$PY" -u train.py \
  --config configs/gen/ss_flow_lora_64_views_canonical_dinov3_50k.json \
  --output_dir "$OUT" \
  --ckpt none \
  --num_gpus 1 \
  --data_dir "$DATA" \
  --auto_retry 3
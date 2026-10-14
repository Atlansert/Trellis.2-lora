#!/usr/bin/env bash
# Stage1 SS LoRA —— **已做 y/z 对齐**的逐视角目标 + DINOv3@512，50k 步。
#
# 目标数据 ss_latents_views_mesh_t1_yz 由 data_toolkit/encode_ss_view_latents_yz.py
# 生成（体素坐标交换 y/z 后重新编码）。实测对齐后 IoU(base, GT) 0.027 → 0.262，
# 与预训练模型/管线约定一致 —— 因此推理时**不需要** --ss_swap_yz。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
PY="/home/test/miniconda3/envs/trellis2/bin/python"
GPU="${1:-5}"
OUT="$ROOT/results/multiview_1186_ss_flow_lora_64_views_alignedyz"
DATA='{"train_data":{"base":"./training_data/multiview_1186","ss_view_latent":"./training_data/multiview_1186/ss_latents_views_mesh_t1_yz","render_cond":"./training_data/multiview_1186/renders_cond"}}'
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$ROOT"
exec "$PY" -u train.py \
  --config configs/gen/ss_flow_lora_64_views_alignedyz_dinov3_50k.json \
  --output_dir "$OUT" --ckpt none --num_gpus 1 --data_dir "$DATA" --auto_retry 3

#!/usr/bin/env bash
# ── 中文摘要 ─────────────────────────────────────────────
# 用途：Stage1（SS flow）LoRA 微调，数据 = training_data/pool1302/train（1302 池的 1146 训练对象）。
# 用法：bash scripts/run_ss_lora_pool1302_canon_axis10.sh [GPU号，默认6]；20k 步、每 2k 存一次、从原厂权重起。
# 输出：results/pool1302_ss_flow_lora_64_canon_axis10/。
# 边界：长跑 GPU 训练；只读 training_data/pool1302/、configs/gen/、checkpoints/；只写输出目录。
# 与旧配方 configs/gen/ss_flow_lora_64_canon_axis10_dinov3_20k.json 完全同参，
# 唯一差异是 canonical_latent_root 指向 pool1302/train/ss_latents_canon。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
PY="/home/test/miniconda3/envs/trellis2/bin/python"
GPU="${1:-6}"
OUT="$ROOT/results/pool1302_ss_flow_lora_64_canon_axis10"
DATA='{"train_data":{"base":"./training_data/pool1302/train","ss_view_latent":"./training_data/pool1302/train/ss_view_latents_index","render_cond":"./training_data/pool1302/train/renders_cond_t"}}'

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$ROOT"
exec "$PY" -u train.py \
  --config configs/gen/ss_flow_lora_64_canon_axis10_pool1302.json \
  --output_dir "$OUT" \
  --ckpt none \
  --num_gpus 1 \
  --data_dir "$DATA" \
  --auto_retry 3

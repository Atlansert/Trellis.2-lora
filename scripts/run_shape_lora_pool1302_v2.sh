#!/usr/bin/env bash
# ── 中文摘要 ─────────────────────────────────────────────
# 用途：Stage2（1024 shape flow）LoRA 微调，数据 = training_data/pool1302/train（1146 对象）。
# 用法：bash scripts/run_shape_lora_pool1302_v2.sh [GPU号，默认7]；v2 配方（r8/a8、只挂 attention、EMA0.9999、lr1e-4）。
# 输出：results/pool1302_shape_flow_lora_1024_v2/。
# 边界：长跑 GPU 训练；只读 training_data/pool1302/、configs/gen/、checkpoints/；只写输出目录。
# 与旧 v2 配方 configs/gen/shape_flow_lora_1024_v2_train.json 完全同参（条件保持 DINOv3/1024），
# 只把 --data_dir 换到 pool1302（shape_latent 与 render_cond 都由 --data_dir 决定，无需改 config）。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
PY="/home/test/miniconda3/envs/trellis2/bin/python"
GPU="${1:-7}"
OUT="$ROOT/results/pool1302_shape_flow_lora_1024_v2"
DATA='{"train_data":{"base":"./training_data/pool1302/train","shape_latent":"./training_data/pool1302/train/shape_latents/shape_enc_next_dc_f16c32_fp16_1024","render_cond":"./training_data/pool1302/train/renders_cond_t"}}'

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$ROOT"
exec "$PY" -u train.py \
  --config configs/gen/shape_flow_lora_1024_v2_pool1302.json \
  --output_dir "$OUT" \
  --ckpt none \
  --num_gpus 1 \
  --data_dir "$DATA" \
  --auto_retry 3

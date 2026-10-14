#!/usr/bin/env bash
# ── 中文摘要 ─────────────────────────────────────────────
# 用途：Stage-2 几何 LoRA 训练（stage2_geometry 配置，配合 Stage1 30000 使用），固定 GPU 7。
# 用法：bash scripts/training_history/run_stage2_geometry_lora_with_stage1_30000.sh（无参数）。
# 输出：results/multiview_1186_stage2_shape_lora_geometry/（checkpoint、log）。
# 边界：长跑 GPU 训练任务，持续写该输出目录；只读 training_data/ 与 configs/。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="/home/test/miniconda3/envs/trellis2/bin/python"
export CUDA_VISIBLE_DEVICES=7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT="$ROOT/results/multiview_1186_stage2_shape_lora_geometry"
DATA='{"train_data":{"base":"./training_data/multiview_1186","shape_latent":"./training_data/multiview_1186/shape_latents/shape_enc_next_dc_f16c32_fp16_1024","render_cond":"./training_data/multiview_1186/renders_cond"}}'
cd "$ROOT"
exec "$PY" train.py \
  --config configs/gen/shape_flow_lora_1024_stage2_geometry_train.json \
  --output_dir "$OUT" \
  --ckpt none \
  --num_gpus 1 \
  --data_dir "$DATA" \
  --auto_retry 3

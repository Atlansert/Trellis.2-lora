#!/usr/bin/env bash
# 规范帧目标 + 只保留目标长轴偏离正侧视轴 <= 10° 的视角。
# 宽高比 >= 3.5 仍包含 30°/45° 斜侧视，那些图和正侧视几乎一样，
# 目标朝向却差几十度，view-side 在 12k 把车身抹成圆管、14k 翻轴。
# 从原厂权重重新训，不续 view-side / B′。每 2k 存一次。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
PY="/home/test/miniconda3/envs/trellis2/bin/python"
GPU="${1:-4}"
OUT="$ROOT/results/multiview_1186_ss_flow_lora_64_canon_axis10"
DATA='{"train_data":{"base":"./training_data/multiview_1186","ss_view_latent":"./training_data/multiview_1186/ss_latents_views_canon","render_cond":"./training_data/multiview_1186/renders_cond_t"}}'

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$ROOT"
exec "$PY" -u train.py \
  --config configs/gen/ss_flow_lora_64_canon_axis10_dinov3_20k.json \
  --output_dir "$OUT" \
  --ckpt none \
  --num_gpus 1 \
  --data_dir "$DATA" \
  --auto_retry 3

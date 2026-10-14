#!/usr/bin/env bash
# 移植 TRELLIS.1 的逐视角方案，按 TRELLIS.2 的底座改了三处（2026-10-12）：
#   1. latent 用修正相机轴之后的逐视角数据 ss_latents_views_canon，每张图配自己的帧
#   2. 条件是 DINOv3@512、crop_scale 1.0，与推理 preprocess_image 一致
#   3. 只保留前景宽高比 >= 3.5 的侧视。端视的长轴不在 index1，会推翻底座先验
# 从原厂权重开始，不续任何已有 LoRA。20k 步，每 2k 存一次。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
PY="/home/test/miniconda3/envs/trellis2/bin/python"
GPU="${1:-7}"
OUT="$ROOT/results/multiview_1186_ss_flow_lora_64_views_side"
DATA='{"train_data":{"base":"./training_data/multiview_1186","ss_view_latent":"./training_data/multiview_1186/ss_latents_views_canon","render_cond":"./training_data/multiview_1186/renders_cond_t"}}'

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$ROOT"
exec "$PY" -u train.py \
  --config configs/gen/ss_flow_lora_64_views_side_dinov3_20k.json \
  --output_dir "$OUT" \
  --ckpt none \
  --num_gpus 1 \
  --data_dir "$DATA" \
  --auto_retry 3

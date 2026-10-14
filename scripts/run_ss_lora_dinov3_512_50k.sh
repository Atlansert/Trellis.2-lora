#!/usr/bin/env bash
# Stage 1 (sparse structure) LoRA —— DINOv3@512 条件 + 与推理对齐的裁剪，50k 步。
#
# 相对旧 run（multiview_1186_ss_flow_lora_64_views）的差异：
#   1. image_cond_model: DinoV2LocalFeatureExtractor@518 → DinoV3FeatureExtractor@512
#      （旧 run 训练用 DINOv2，推理管线用 DINOv3，条件特征空间不一致，
#        实测 SS occupancy IoU 0.525 → 0.054）
#   2. dataset.crop_scale: 1.2 → 1.0（物体占满画面，与推理 preprocess_image 一致）
#   3. dataset.image_size: 518 → 512（与 DINOv3 输入分辨率一致）
#   4. snapshot_sampler_params 对齐线上 pipeline.json 的 SS 采样参数
#   5. max_steps 50000、i_sample 5000、i_save 1000
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
PY="/home/test/miniconda3/envs/trellis2/bin/python"
GPU="${1:-4}"
OUT="$ROOT/results/multiview_1186_ss_flow_lora_64_views_dinov3_512"
DATA='{"train_data":{"base":"./training_data/multiview_1186","ss_view_latent":"./training_data/multiview_1186/ss_latents_views_mesh_t1","render_cond":"./training_data/multiview_1186/renders_cond"}}'

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$ROOT"
exec "$PY" -u train.py \
  --config configs/gen/ss_flow_lora_64_views_dinov3_512_50k.json \
  --output_dir "$OUT" \
  --ckpt none \
  --num_gpus 1 \
  --data_dir "$DATA" \
  --auto_retry 3
#!/usr/bin/env bash
# ── 中文摘要 ─────────────────────────────────────────────
# 用途：Stage1（SS）LoRA 训练，用"逐对象朝向规范化后"的数据 ss_latents_views_canon + renders_cond_t，50k 步。
# 用法：bash scripts/run_ss_lora_canonv2_dinov3_50k.sh [GPU 号，默认 4]。
# 输出：results/multiview_1186_ss_flow_lora_64_views_canonv2/。
# 边界：长跑 GPU 训练任务，持续写该输出目录；口径与 A run 一致（DINOv3@512、crop 1.0、r8/a8）以便对比。
# Stage1 SS LoRA —— **逐对象朝向规范化后的数据**（治本方案，2026-10-11）。
#
# 数据来源 training_data/multiview_1186/ss_latents_views_canon：
#   由 data_toolkit/generate_ss_view_voxels_canon.py（修正相机环绕轴 + 手性）
#   → encode_ss_view_latents_canon.py（轴置换对齐管线约定）生成。
#   关键差别：旧数据的体素化绕错了轴，占 71% 的对象 latent 与视角无关（图文不匹配）。
#
# 其余口径与本轮 A run 一致，便于对比：
#   DINOv3@512 条件、crop_scale 1.0、image_size 512、50k 步、LoRA r8/a8。
# 条件图改用 renders_cond_t（统一透明），与推理 preprocess_image 口径一致。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
PY="/home/test/miniconda3/envs/trellis2/bin/python"
GPU="${1:-4}"
OUT="$ROOT/results/multiview_1186_ss_flow_lora_64_views_canonv2"
DATA='{"train_data":{"base":"./training_data/multiview_1186","ss_view_latent":"./training_data/multiview_1186/ss_latents_views_canon","render_cond":"./training_data/multiview_1186/renders_cond_t"}}'

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$ROOT"
exec "$PY" -u train.py \
  --config configs/gen/ss_flow_lora_64_views_canonv2_dinov3_50k.json \
  --output_dir "$OUT" \
  --ckpt none \
  --num_gpus 1 \
  --data_dir "$DATA" \
  --auto_retry 3
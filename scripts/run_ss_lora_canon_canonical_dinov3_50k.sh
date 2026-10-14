#!/usr/bin/env bash
# Stage1 SS LoRA —— **修正后的规范帧（canonical）目标**，50k 步（2026-10-11）。
#
# 为什么改用规范帧：实测原厂 base 在**端视**输入上也输出长度轴 = index1
# （见 scripts/utils/canon_axis_agreement.py 的输出），说明预训练模型的约定是
# 规范帧、不是相机帧。逐视角（相机帧）监督等于让模型"随视角改变长度轴"，
# 是在对抗预训练先验 —— 这正是此前所有 run 摇摆的深层原因。
#
# 目标数据 ss_latents_canon 由 data_toolkit/build_canon_canonical_latents.py 生成：
# 从**已修正相机约定**的逐视角 latent 中取"最侧视"那一帧作为该对象的规范帧目标，
# 因此在侧视帧上长度轴天然落在 index1，全部对象共享同一轴约定。
#
# 条件图用 renders_cond_t（统一透明，与推理口径一致）。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
PY="/home/test/miniconda3/envs/trellis2/bin/python"
GPU="${1:-5}"
OUT="$ROOT/results/multiview_1186_ss_flow_lora_64_views_canon_canonical"
DATA='{"train_data":{"base":"./training_data/multiview_1186","ss_view_latent":"./training_data/multiview_1186/ss_latents_views_canon","render_cond":"./training_data/multiview_1186/renders_cond_t"}}'

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$ROOT"
exec "$PY" -u train.py \
  --config configs/gen/ss_flow_lora_64_views_canon_canonical_dinov3_50k.json \
  --output_dir "$OUT" \
  --ckpt none \
  --num_gpus 1 \
  --data_dir "$DATA" \
  --auto_retry 3
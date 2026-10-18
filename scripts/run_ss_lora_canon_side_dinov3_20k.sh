#!/usr/bin/env bash
# ── 中文摘要 ─────────────────────────────────────────────
# 用途：Stage1（SS）LoRA 训练，规范帧目标但条件图只保留侧视（前景宽高比 ≥3.5），从原厂权重重训。
# 用法：bash scripts/run_ss_lora_canon_side_dinov3_20k.sh [GPU 号，默认 7]（步数以所引 config json 为准）。
# 输出：results/multiview_1186_ss_flow_lora_64_views_canon_side/。
# 边界：长跑 GPU 训练任务，持续写该输出目录；验收看成品长高比而非 loss，不开 canonicalize。
# Stage1 SS LoRA —— 规范帧目标，但条件图只保留侧视（2026-10-11）。
#
# B′（canon_canonical，全 24 视角）把端视和侧视配成同一个满格侧视 latent。
# 端视的长高比和目标相反，模型不再读输入图的比例，10k→24k 把 001 从
# 长高比 8.0 拉到 8.9（输入图 4.22，训练集目标中位只有 5.33）。
#
# 本轮只保留前景宽高比 >= 3.5 的视角（dataset.min_view_aspect）。
# 从原厂 SS 权重重新开始，不从 B′@24k 续：那个 LoRA 已经是细管子。
# 20k 步、每 2k 存一次。验收看成品长高比 / 输入宽高比，不看 loss，不开 canonicalize。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
PY="/home/test/miniconda3/envs/trellis2/bin/python"
GPU="${1:-7}"
OUT="$ROOT/results/multiview_1186_ss_flow_lora_64_views_canon_side"
DATA='{"train_data":{"base":"./training_data/multiview_1186","ss_view_latent":"./training_data/multiview_1186/ss_latents_views_canon","render_cond":"./training_data/multiview_1186/renders_cond_t"}}'

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$ROOT"
exec "$PY" -u train.py \
  --config configs/gen/ss_flow_lora_64_views_canon_side_dinov3_50k.json \
  --output_dir "$OUT" \
  --ckpt none \
  --num_gpus 1 \
  --data_dir "$DATA" \
  --auto_retry 3

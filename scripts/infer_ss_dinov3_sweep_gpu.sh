#!/usr/bin/env bash
# ── 中文摘要 ─────────────────────────────────────────────
# 用途：推理扫描的单档任务：shape flow 固定 20k，SS LoRA（DINOv3@512 重训版）取指定步数，只跑 data_test/001.png。
# 用法：bash scripts/infer_ss_dinov3_sweep_gpu.sh <ss_step> <gpu>（步数补零到 7 位拼权重名）。
# 输出：results_infer/sweep_ss_dinov3_<step>_shape20k/（001.glb、run.log、infer.log）；已有 001.glb 则跳过。
# 边界：会 mkdir 输出目录并调用 infer_finetuned.py（GPU 推理、--no_video）；权重缺失退 2、无输出 GLB 退 3。
# Task 4 推理扫描：shape flow 固定 20k，SS LoRA（DINOv3@512 重训版）从 10k 到 50k 每 10k 一档，
# 对 data_test/001.png 各推理一次。
#
# Usage: $0 <ss_step> <gpu>
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
PY="/home/test/miniconda3/envs/trellis2/bin/python"
SSN="$1"; GPU="$2"

SS_DIR="$ROOT/results/multiview_1186_ss_flow_lora_64_views_dinov3_512/ckpts"
SS_CKPT="$SS_DIR/denoiser_ema0.9999_step$(printf '%07d' "$SSN").pt"
SF_CKPT="$ROOT/results/multiview_1186_flow_1024_lora_v2/ckpts/denoiser_ema0.9999_step0020000.pt"
DEC="$ROOT/checkpoints/ckpts/shape_dec_next_dc_f16c32_fp16.pt"
INPUT="$ROOT/data_test/001.png"
OUT="$ROOT/results_infer/sweep_ss_dinov3_$(printf '%07d' "$SSN")_shape20k"
LOG="$OUT/run.log"

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT"

if [[ ! -f "$SS_CKPT" ]]; then
    echo "[$(date -Is)] SS 检查点不存在: $SS_CKPT" | tee -a "$LOG"; exit 2
fi
if [[ -s "$OUT/001.glb" ]]; then
    echo "[$(date -Is)] ss=$SSN 已完成，跳过" | tee -a "$LOG"; exit 0
fi
echo "[$(date -Is)] start ss=$SSN gpu=$GPU" | tee -a "$LOG"
"$PY" -u scripts/infer_finetuned.py \
  --image "$INPUT" --output "$OUT" \
  --pipeline_type 1024 --vae_decoder "$DEC" --vae_only \
  --sparse_structure_lora "$SS_CKPT" --ss_lora_rank 8 --ss_lora_alpha 8 \
  --flow_1024_lora "$SF_CKPT" --shape_lora_rank 8 --shape_lora_alpha 8 \
  --no_video >> "$OUT/infer.log" 2>&1
rc=$?
if [[ -s "$OUT/001.glb" ]]; then
    sz=$(stat -c%s "$OUT/001.glb")
    echo "[$(date -Is)] finish ss=$SSN glb=$sz bytes" | tee -a "$LOG"
else
    echo "[$(date -Is)] ERROR ss=$SSN rc=$rc 无输出 GLB" | tee -a "$LOG"; exit 3
fi
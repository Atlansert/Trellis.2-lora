#!/usr/bin/env bash
# 通用推理扫描：<SS LoRA 所在目录> 的 10k..50k，每 10k 一档，shape flow 固定 20k。
#
# Usage: infer_ss_sweep_generic.sh <ckpt_dir> <step> <gpu> <tag>
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
PY="/home/test/miniconda3/envs/trellis2/bin/python"
CKDIR="$1"; SSN="$2"; GPU="$3"; TAG="$4"

SS_CKPT="$CKDIR/denoiser_ema0.9999_step$(printf '%07d' "$SSN").pt"
SF_CKPT="$ROOT/results/multiview_1186_flow_1024_lora_v2/ckpts/denoiser_ema0.9999_step0020000.pt"
DEC="$ROOT/checkpoints/ckpts/shape_dec_next_dc_f16c32_fp16.pt"
OUT="$ROOT/results_infer/sweep_${TAG}_$(printf '%07d' "$SSN")_shape20k"
LOG="$OUT/run.log"

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT"
if [[ ! -f "$SS_CKPT" ]]; then
    echo "[$(date -Is)] SS 检查点不存在: $SS_CKPT" | tee -a "$LOG"; exit 2
fi
if [[ -s "$OUT/001.glb" ]]; then
    echo "[$(date -Is)] $TAG ss=$SSN 已完成，跳过" | tee -a "$LOG"; exit 0
fi
echo "[$(date -Is)] start $TAG ss=$SSN gpu=$GPU" | tee -a "$LOG"
"$PY" -u scripts/infer_finetuned.py \
  --image "$ROOT/data_test/001.png" --output "$OUT" \
  --pipeline_type 1024 --vae_decoder "$DEC" --vae_only \
  --sparse_structure_lora "$SS_CKPT" --ss_lora_rank 8 --ss_lora_alpha 8 \
  --flow_1024_lora "$SF_CKPT" --shape_lora_rank 8 --shape_lora_alpha 8 \
  --no_video >> "$OUT/infer.log" 2>&1
if [[ -s "$OUT/001.glb" ]]; then
    echo "[$(date -Is)] finish $TAG ss=$SSN glb=$(stat -c%s "$OUT/001.glb") bytes" | tee -a "$LOG"
else
    echo "[$(date -Is)] ERROR $TAG ss=$SSN 无输出 GLB" | tee -a "$LOG"; exit 3
fi
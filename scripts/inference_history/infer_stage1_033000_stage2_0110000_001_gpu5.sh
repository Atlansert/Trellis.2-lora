#!/usr/bin/env bash
# ── 中文摘要 ─────────────────────────────────────────────
# 用途：单图 data_test/001.png 推理：Stage1 SS LoRA 33000 + Stage2 shape LoRA 110000，固定 GPU 5。
# 用法：bash scripts/inference_history/infer_stage1_033000_stage2_0110000_001_gpu5.sh（无参数）。
# 输出：results_infer/data_test_stage1_lora_033000_stage2_lora_0110000_001/（001.glb、run.log、infer.log）。
# 边界：会 mkdir 输出目录并跑 GPU 推理（--no_video）；没产出 001.glb 则退出码 3。
# Inference: Stage1 SS LoRA step 33000 + Stage2 shape-flow LoRA step 110000
# on data_test/001.png only, GPU 5.
# Output: results_infer/data_test_stage1_lora_033000_stage2_lora_0110000_001/
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
PY="/home/test/miniconda3/envs/trellis2/bin/python"
SS_CKPT="$ROOT/results/multiview_1186_ss_flow_lora_64_views/ckpts/denoiser_ema0.9999_step0033000.pt"
SF_CKPT="$ROOT/results/multiview_1186_flow_1024_lora_v2/ckpts/denoiser_ema0.9999_step0110000.pt"
DEC="$ROOT/checkpoints/ckpts/shape_dec_next_dc_f16c32_fp16.pt"
IMAGE="$ROOT/data_test/001.png"
OUT="$ROOT/results_infer/data_test_stage1_lora_033000_stage2_lora_0110000_001"
LOG="$OUT/run.log"

export CUDA_VISIBLE_DEVICES=5
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT"
cd "$ROOT"

if [[ ! -s "$SS_CKPT" || ! -s "$SF_CKPT" || ! -f "$IMAGE" ]]; then
    echo "[$(date -Is)] ERROR missing ckpts or image" | tee -a "$LOG"
    exit 2
fi
if [[ -s "$OUT/001.glb" ]]; then
    echo "[$(date -Is)] 001.glb already exists, skip" | tee -a "$LOG"
    exit 0
fi
echo "[$(date -Is)] start ss=$SS_CKPT sf=$SF_CKPT img=001" | tee -a "$LOG"
"$PY" -u "$ROOT/scripts/infer_finetuned.py" \
  --image "$IMAGE" \
  --output "$OUT" \
  --pipeline_type 1024 \
  --vae_decoder "$DEC" \
  --vae_only \
  --sparse_structure_lora "$SS_CKPT" \
  --ss_lora_rank 8 \
  --ss_lora_alpha 8 \
  --flow_1024_lora "$SF_CKPT" \
  --shape_lora_rank 8 \
  --shape_lora_alpha 8 \
  --no_video \
  >> "$OUT/infer.log" 2>&1
if [[ -s "$OUT/001.glb" ]]; then
    echo "[$(date -Is)] finish OK" | tee -a "$LOG"
else
    echo "[$(date -Is)] ERROR: 001.glb missing" | tee -a "$LOG"
    exit 3
fi
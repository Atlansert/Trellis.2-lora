#!/usr/bin/env bash
# Inference: Stage1 SS LoRA step 32000 + Stage2 pretrained (no LoRA), 001 only, GPU 5.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
PY="/home/test/miniconda3/envs/trellis2/bin/python"
SS_CKPT="$ROOT/results/multiview_1186_ss_flow_lora_64_views/ckpts/denoiser_ema0.9999_step0032000.pt"
DEC="$ROOT/checkpoints/ckpts/shape_dec_next_dc_f16c32_fp16.pt"
IMAGE="$ROOT/data_test/001.png"
OUT="$ROOT/results_infer/data_test_stage1_lora_032000_stage2_pretrained_1024"
LOG="$OUT/run.log"
export CUDA_VISIBLE_DEVICES=5
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT"
cd "$ROOT"
if [[ ! -s "$SS_CKPT" ]]; then
    echo "[$(date -Is)] ERROR missing stage1 ckpt" | tee -a "$LOG"
    exit 2
fi
echo "[$(date -Is)] start ss=$SS_CKPT (stage2 pretrained)" | tee -a "$LOG"
"$PY" -u scripts/infer_finetuned.py \
  --image "$IMAGE" --output "$OUT" \
  --pipeline_type 1024 --vae_decoder "$DEC" --vae_only \
  --sparse_structure_lora "$SS_CKPT" --ss_lora_rank 8 --ss_lora_alpha 8 \
  --no_video >> "$OUT/infer.log" 2>&1
if [[ -s "$OUT/001.glb" ]]; then
    echo "[$(date -Is)] finish OK" | tee -a "$LOG"
else
    echo "[$(date -Is)] ERROR 001.glb missing" | tee -a "$LOG"; exit 3
fi

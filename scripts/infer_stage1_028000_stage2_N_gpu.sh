#!/usr/bin/env bash
# Usage: $0 <stage2_step> <gpu>   (stage1 fixed at 28000, full data_test)
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
PY="/home/test/miniconda3/envs/trellis2/bin/python"
S2="$1"; GPU="$2"
SS_CKPT="$ROOT/results/multiview_1186_ss_flow_lora_64_views/ckpts/denoiser_ema0.9999_step0028000.pt"
SF_CKPT="$ROOT/results/multiview_1186_flow_1024_lora_v2/ckpts/denoiser_ema0.9999_step$(printf '%07d' "$S2").pt"
DEC="$ROOT/checkpoints/ckpts/shape_dec_next_dc_f16c32_fp16.pt"
INPUT="$ROOT/data_test"
OUT="$ROOT/results_infer/data_test_stage1_lora_028000_stage2_lora_$(printf '%07d' "$S2")"
LOG="$OUT/run.log"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT"
cd "$ROOT"
nimgs=$(find "$INPUT" -maxdepth 1 -type f \( -name '*.png' -o -name '*.PNG' -o -name '*.jpg' -o -name '*.jpeg' -o -name '*.webp' \) | wc -l)
count=$(find "$OUT" -maxdepth 1 -type f -name '*.glb' -size +0c | wc -l)
if [[ "$count" -eq "$nimgs" ]]; then
    echo "[$(date -Is)] already complete ($nimgs GLBs), skip" | tee -a "$LOG"; exit 0
fi
echo "[$(date -Is)] start s2=$S2 gpu=$GPU" | tee -a "$LOG"
"$PY" -u scripts/infer_finetuned.py \
  --input_dir "$INPUT" --output "$OUT" \
  --pipeline_type 1024 --vae_decoder "$DEC" --vae_only \
  --sparse_structure_lora "$SS_CKPT" --ss_lora_rank 8 --ss_lora_alpha 8 \
  --flow_1024_lora "$SF_CKPT" --shape_lora_rank 8 --shape_lora_alpha 8 \
  --no_video >> "$OUT/infer.log" 2>&1
count=$(find "$OUT" -maxdepth 1 -type f -name '*.glb' -size +0c | wc -l)
echo "[$(date -Is)] finish glbs=$count/$nimgs" | tee -a "$LOG"
[[ "$count" -eq "$nimgs" ]] || { echo "[$(date -Is)] ERROR incomplete" | tee -a "$LOG"; exit 3; }

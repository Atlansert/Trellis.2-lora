#!/usr/bin/env bash
set -euo pipefail

# Resolve the physical workspace path.  The workspace is also mounted at
# /home/lyx/lzx/TRELLIS.2; using `pwd` here can send logs/results to the
# alternate logical path and makes progress appear missing under /data/lzx.
ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
PY="/home/test/miniconda3/envs/trellis2/bin/python"
CKPT_DIR="$ROOT/results/multiview_1186_ss_flow_lora_64_views/ckpts"
DEC="$ROOT/checkpoints/ckpts/shape_dec_next_dc_f16c32_fp16.pt"
INPUT="$ROOT/data_test"
BASE_OUT="$ROOT/results_infer/data_test_stage1_lora_stage2_pretrained_1024"
LOG="$BASE_OUT/run.log"

export CUDA_VISIBLE_DEVICES=7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$BASE_OUT"
cd "$ROOT"

# Requested evaluation range: 031000 through 039000 (increments of 1,000).
# Keep this explicit so 031000 is never confused with 0310000.
for n in $(seq 31000 1000 39000); do
    step=$(printf '%07d' "$n")
    ckpt="$CKPT_DIR/denoiser_ema0.9999_step${step}.pt"
    out="$BASE_OUT/step$(printf '%06d' "$n")"
    if [[ ! -s "$ckpt" ]]; then
        echo "[$(date -Is)] ERROR missing checkpoint: $ckpt" | tee -a "$LOG"
        exit 2
    fi
    mkdir -p "$out"
    # A complete step has one GLB for each of the eight data_test images.
    count=$(find "$out" -maxdepth 1 -type f -name '*.glb' -size +0c | wc -l)
    if [[ "$count" -eq 8 ]]; then
        echo "[$(date -Is)] skip step=$n (8 GLBs already present)" | tee -a "$LOG"
        continue
    fi
    echo "[$(date -Is)] start step=$n checkpoint=$ckpt" | tee -a "$LOG"
    "$PY" -u scripts/infer_finetuned.py \
      --input_dir "$INPUT" \
      --output "$out" \
      --pipeline_type 1024 \
      --vae_decoder "$DEC" \
      --vae_only \
      --sparse_structure_lora "$ckpt" \
      --ss_lora_rank 8 \
      --ss_lora_alpha 8 \
      --no_video \
      >> "$out/infer.log" 2>&1
    count=$(find "$out" -maxdepth 1 -type f -name '*.glb' -size +0c | wc -l)
    echo "[$(date -Is)] finish step=$n glbs=$count" | tee -a "$LOG"
    if [[ "$count" -ne 8 ]]; then
        echo "[$(date -Is)] ERROR incomplete step=$n, see $out/infer.log" | tee -a "$LOG"
        exit 3
    fi
done
echo "[$(date -Is)] all steps finished" | tee -a "$LOG"

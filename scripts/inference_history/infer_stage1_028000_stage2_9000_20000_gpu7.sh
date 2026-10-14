#!/usr/bin/env bash
# Combined inference: Stage-1 SS LoRA (step 28000) + Stage-2 shape-flow LoRA
# (EMA steps 9000..20000) on data_test, single GPU (7), sequential.
# Output: results_infer/data_test_stage1_lora_028000_stage2_lora_1024_9000_20000/step0NNNNN
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
PY="/home/test/miniconda3/envs/trellis2/bin/python"
SS_CKPT="$ROOT/results/multiview_1186_ss_flow_lora_64_views/ckpts/denoiser_ema0.9999_step0028000.pt"
SF_DIR="$ROOT/results/multiview_1186_flow_1024_lora_v2/ckpts"
DEC="$ROOT/checkpoints/ckpts/shape_dec_next_dc_f16c32_fp16.pt"
INPUT="$ROOT/data_test"
BASE_OUT="$ROOT/results_infer/data_test_stage1_lora_028000_stage2_lora_1024_9000_20000"
LOG="$BASE_OUT/run.log"

export CUDA_VISIBLE_DEVICES=7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$BASE_OUT"
cd "$ROOT"

if [[ ! -s "$SS_CKPT" ]]; then
    echo "[$(date -Is)] ERROR missing stage1 ckpt: $SS_CKPT" | tee -a "$LOG"
    exit 2
fi

for n in $(seq 9000 1000 20000); do
    sf_ckpt="$SF_DIR/denoiser_ema0.9999_step$(printf '%07d' "$n").pt"
    out="$BASE_OUT/step$(printf '%06d' "$n")"
    if [[ ! -s "$sf_ckpt" ]]; then
        echo "[$(date -Is)] ERROR missing stage2 ckpt: $sf_ckpt" | tee -a "$LOG"
        exit 2
    fi
    mkdir -p "$out"
    count=$(find "$out" -maxdepth 1 -type f -name '*.glb' -size +0c | wc -l)
    if [[ "$count" -eq 8 ]]; then
        echo "[$(date -Is)] skip step=$n (8 GLBs already present)" | tee -a "$LOG"
        continue
    fi
    echo "[$(date -Is)] start step=$n ss=$SS_CKPT sf=$sf_ckpt" | tee -a "$LOG"
    "$PY" -u "$ROOT/scripts/infer_finetuned.py" \
      --input_dir "$INPUT" \
      --output "$out" \
      --pipeline_type 1024 \
      --vae_decoder "$DEC" \
      --vae_only \
      --sparse_structure_lora "$SS_CKPT" \
      --ss_lora_rank 8 \
      --ss_lora_alpha 8 \
      --flow_1024_lora "$sf_ckpt" \
      --shape_lora_rank 8 \
      --shape_lora_alpha 8 \
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
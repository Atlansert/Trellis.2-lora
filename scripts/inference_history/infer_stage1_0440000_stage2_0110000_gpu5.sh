#!/usr/bin/env bash
# ── 中文摘要 ─────────────────────────────────────────────
# 用途：data_test 全量（8 张）推理：Stage1 SS LoRA 440000 + Stage2 shape LoRA 110000，固定 GPU 5。
# 用法：bash scripts/inference_history/infer_stage1_0440000_stage2_0110000_gpu5.sh（无参数）。
# 输出：results_infer/data_test_stage1_lora_0440000_stage2_lora_0110000/（GLB、run.log、infer.log）；已有等量 GLB 则跳过。
# 边界：会 mkdir 输出目录并跑 GPU 推理（--no_video）；跑完 GLB 数不足则退出码 3。
# Inference: Stage1 SS LoRA step 440000 + Stage2 shape-flow LoRA step 110000
# on full data_test (8 images), GPU 5.
# Output: results_infer/data_test_stage1_lora_0440000_stage2_lora_0110000/
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
PY="/home/test/miniconda3/envs/trellis2/bin/python"
SS_CKPT="$ROOT/results/multiview_1186_ss_flow_lora_64_views/ckpts/denoiser_ema0.9999_step0440000.pt"
SF_CKPT="$ROOT/results/multiview_1186_flow_1024_lora_v2/ckpts/denoiser_ema0.9999_step0110000.pt"
DEC="$ROOT/checkpoints/ckpts/shape_dec_next_dc_f16c32_fp16.pt"
INPUT="$ROOT/data_test"
OUT="$ROOT/results_infer/data_test_stage1_lora_0440000_stage2_lora_0110000"
LOG="$OUT/run.log"

export CUDA_VISIBLE_DEVICES=5
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT"
cd "$ROOT"

if [[ ! -s "$SS_CKPT" || ! -s "$SF_CKPT" ]]; then
    echo "[$(date -Is)] ERROR missing ckpts" | tee -a "$LOG"
    exit 2
fi
count=$(find "$OUT" -maxdepth 1 -type f -name '*.glb' -size +0c | wc -l)
if [[ "$count" -eq 8 ]]; then
    echo "[$(date -Is)] already complete (8 GLBs), skip" | tee -a "$LOG"
    exit 0
fi
echo "[$(date -Is)] start ss=$SS_CKPT sf=$SF_CKPT" | tee -a "$LOG"
"$PY" -u "$ROOT/scripts/infer_finetuned.py" \
  --input_dir "$INPUT" \
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
count=$(find "$OUT" -maxdepth 1 -type f -name '*.glb' -size +0c | wc -l)
echo "[$(date -Is)] finish glbs=$count" | tee -a "$LOG"
[[ "$count" -eq 8 ]] || { echo "[$(date -Is)] ERROR incomplete: $count/8" | tee -a "$LOG"; exit 3; }
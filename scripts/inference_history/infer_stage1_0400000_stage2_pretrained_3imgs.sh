#!/usr/bin/env bash
# ── 中文摘要 ─────────────────────────────────────────────
# 用途：Stage1 SS LoRA 400000 + Stage2 保持预训练（不挂 shape LoRA）推理，输入 data_test_sel（001/003/005_side），固定 GPU 5。
# 用法：bash scripts/inference_history/infer_stage1_0400000_stage2_pretrained_3imgs.sh（无参数）。
# 输出：results_infer/data_test_stage1_lora_0400000_stage2_pretrained_1024/（GLB、run.log、infer.log）。
# 边界：会 mkdir 输出目录并跑 GPU 推理（--no_video）；跑完 GLB 数不足则退出码 3。
# Inference: Stage1 SS LoRA (step 400000) + Stage2 pretrained (no LoRA)
# on selected data_test images (001, 003, 005_side), GPU 5.
# Output: results_infer/data_test_stage1_lora_0400000_stage2_pretrained_1024/
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
PY="/home/test/miniconda3/envs/trellis2/bin/python"
SS_CKPT="$ROOT/results/multiview_1186_ss_flow_lora_64_views/ckpts/denoiser_ema0.9999_step0400000.pt"
DEC="$ROOT/checkpoints/ckpts/shape_dec_next_dc_f16c32_fp16.pt"
INPUT="$ROOT/data_test_sel"
OUT="$ROOT/results_infer/data_test_stage1_lora_0400000_stage2_pretrained_1024"
LOG="$OUT/run.log"

export CUDA_VISIBLE_DEVICES=5
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT"
cd "$ROOT"

if [[ ! -s "$SS_CKPT" ]]; then
    echo "[$(date -Is)] ERROR missing stage1 ckpt: $SS_CKPT" | tee -a "$LOG"
    exit 2
fi
count=$(find "$OUT" -maxdepth 1 -type f -name '*.glb' -size +0c | wc -l)
if [[ "$count" -eq 3 ]]; then
    echo "[$(date -Is)] already complete (3 GLBs), skip" | tee -a "$LOG"
    exit 0
fi
echo "[$(date -Is)] start ss=$SS_CKPT" | tee -a "$LOG"
"$PY" -u "$ROOT/scripts/infer_finetuned.py" \
  --input_dir "$INPUT" \
  --output "$OUT" \
  --pipeline_type 1024 \
  --vae_decoder "$DEC" \
  --vae_only \
  --sparse_structure_lora "$SS_CKPT" \
  --ss_lora_rank 8 \
  --ss_lora_alpha 8 \
  --no_video \
  >> "$OUT/infer.log" 2>&1
count=$(find "$OUT" -maxdepth 1 -type f -name '*.glb' -size +0c | wc -l)
echo "[$(date -Is)] finish glbs=$count" | tee -a "$LOG"
[[ "$count" -eq 3 ]] || { echo "[$(date -Is)] ERROR incomplete: $count/3" | tee -a "$LOG"; exit 3; }
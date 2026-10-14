#!/usr/bin/env bash
# ── 中文摘要 ─────────────────────────────────────────────
# 用途：补跑 Stage1(28k) + Stage2(20k) 的 step020000 这一档，把结果补进统一目录，固定 GPU 7。
# 用法：bash scripts/inference_history/infer_28k_stage2_20000_complete_gpu7.sh（无参数）。
# 输出：results_infer/data_test_stage1_lora_028000_stage2_lora_1024/step020000/（GLB、infer.log），日志在上级 run.log。
# 边界：会 mkdir 输出目录并跑 GPU 推理（--no_video）；已满 8 个 GLB 则跳过，不足 8 个退出码 3。
# Complete step20000 for Stage1(28k) + Stage2(20k) into the unified results dir.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
PY="/home/test/miniconda3/envs/trellis2/bin/python"
SS_CKPT="$ROOT/results/multiview_1186_ss_flow_lora_64_views/ckpts/denoiser_ema0.9999_step0028000.pt"
SF_CKPT="$ROOT/results/multiview_1186_flow_1024_lora_v2/ckpts/denoiser_ema0.9999_step0020000.pt"
DEC="$ROOT/checkpoints/ckpts/shape_dec_next_dc_f16c32_fp16.pt"
INPUT="$ROOT/data_test"
OUT="$ROOT/results_infer/data_test_stage1_lora_028000_stage2_lora_1024/step020000"
LOG="$ROOT/results_infer/data_test_stage1_lora_028000_stage2_lora_1024/run.log"

export CUDA_VISIBLE_DEVICES=7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT"
cd "$ROOT"
count=$(find "$OUT" -maxdepth 1 -type f -name '*.glb' -size +0c | wc -l)
if [[ "$count" -eq 8 ]]; then
    echo "[$(date -Is)] step=20000 already complete (8 GLBs)" | tee -a "$LOG"
    exit 0
fi
echo "[$(date -Is)] start step=20000 ss=$SS_CKPT sf=$SF_CKPT" | tee -a "$LOG"
"$PY" -u "$ROOT/scripts/infer_finetuned.py" \
  --input_dir "$INPUT" --output "$OUT" \
  --pipeline_type 1024 --vae_decoder "$DEC" --vae_only \
  --sparse_structure_lora "$SS_CKPT" --ss_lora_rank 8 --ss_lora_alpha 8 \
  --flow_1024_lora "$SF_CKPT" --shape_lora_rank 8 --shape_lora_alpha 8 \
  --no_video >> "$OUT/infer.log" 2>&1
count=$(find "$OUT" -maxdepth 1 -type f -name '*.glb' -size +0c | wc -l)
echo "[$(date -Is)] finish step=20000 glbs=$count" | tee -a "$LOG"
[[ "$count" -eq 8 ]] || { echo "[$(date -Is)] ERROR incomplete step=20000" | tee -a "$LOG"; exit 3; }
echo "[$(date -Is)] step=20000 complete" | tee -a "$LOG"

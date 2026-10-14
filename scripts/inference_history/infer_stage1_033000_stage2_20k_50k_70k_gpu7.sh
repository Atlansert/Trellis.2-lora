#!/usr/bin/env bash
# ── 中文摘要 ─────────────────────────────────────────────
# 用途：Stage1 SS LoRA 固定 33000，Stage2 shape LoRA 依次跑 20000/50000/70000 三档，对 data_test 全量串行推理，GPU 7。
# 用法：bash scripts/inference_history/infer_stage1_033000_stage2_20k_50k_70k_gpu7.sh（无参数）。
# 输出：results_infer/data_test_stage1_lora_033000_stage2_lora_<step>/（GLB、run.log、infer.log）；每档已有等量 GLB 则跳过。
# 边界：会 mkdir 输出目录并跑 GPU 推理（--no_video）；缺 stage2 权重退 2，某档 GLB 不足退 3。
# Stage1 SS LoRA step 33000 + Stage2 shape-flow LoRA steps 20000/50000/70000
# sequential inference on full data_test (8 images), GPU 7.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
PY="/home/test/miniconda3/envs/trellis2/bin/python"
SS_CKPT="$ROOT/results/multiview_1186_ss_flow_lora_64_views/ckpts/denoiser_ema0.9999_step0033000.pt"
SF_DIR="$ROOT/results/multiview_1186_flow_1024_lora_v2/ckpts"
DEC="$ROOT/checkpoints/ckpts/shape_dec_next_dc_f16c32_fp16.pt"
INPUT="$ROOT/data_test"

export CUDA_VISIBLE_DEVICES=7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for n in 20000 50000 70000; do
    sf_ckpt="$SF_DIR/denoiser_ema0.9999_step$(printf '%07d' "$n").pt"
    OUT="$ROOT/results_infer/data_test_stage1_lora_033000_stage2_lora_$(printf '%07d' "$n")"
    LOG="$OUT/run.log"
    mkdir -p "$OUT"
    cd "$ROOT"
    if [[ ! -s "$sf_ckpt" ]]; then
        echo "[$(date -Is)] ERROR missing stage2 ckpt: $sf_ckpt" | tee -a "$LOG"
        exit 2
    fi
    nimgs=$(find "$INPUT" -maxdepth 1 -type f \( -name '*.png' -o -name '*.PNG' -o -name '*.jpg' -o -name '*.jpeg' -o -name '*.webp' \) | wc -l)
    count=$(find "$OUT" -maxdepth 1 -type f -name '*.glb' -size +0c | wc -l)
    if [[ "$count" -eq "$nimgs" ]]; then
        echo "[$(date -Is)] step=$n already complete (8 GLBs), skip" | tee -a "$LOG"
        continue
    fi
    echo "[$(date -Is)] start sf=$n ss=$SS_CKPT" | tee -a "$LOG"
    "$PY" -u "$ROOT/scripts/infer_finetuned.py" \
      --input_dir "$INPUT" \
      --output "$OUT" \
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
      >> "$OUT/infer.log" 2>&1
    count=$(find "$OUT" -maxdepth 1 -type f -name '*.glb' -size +0c | wc -l)
    echo "[$(date -Is)] finish sf=$n glbs=$count" | tee -a "$LOG"
    [[ "$count" -eq "$nimgs" ]] || { echo "[$(date -Is)] ERROR incomplete $n: $count/$nimgs" | tee -a "$LOG"; exit 3; }
done
echo "[$(date -Is)] all three step inferences finished" | tee -a "$ROOT/results_infer/data_test_stage1_lora_033000_stage2_lora_0002000/run.log"
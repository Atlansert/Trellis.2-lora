#!/usr/bin/env bash
# ── 中文摘要 ─────────────────────────────────────────────
# 用途：Stage1 SS LoRA 的 EMA 20000..29000 各步数并行推理（轮转分配到 4/5/6/7 四张卡），Stage2 保持预训练。
# 用法：bash scripts/inference_history/infer_data_test_stage1_lora_20k_29k_parallel.sh（无参数）。
# 输出：results_infer/data_test_stage1_lora_stage2_pretrained_1024_20k_29k/step0NNNNN/（GLB、infer.log），总日志 run.log。
# 边界：会 mkdir 输出目录并起多个 GPU 推理进程；某档权重缺失只记错并返回失败，不影响其它档。
# Parallel Stage-1 LoRA inference for EMA steps 20000..29000 on data_test.
# Distributes checkpoints round-robin across the four free GPUs (4,5,6,7).
# Output: results_infer/data_test_stage1_lora_stage2_pretrained_1024_20k_29k/step0NNNNN
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
PY="/home/test/miniconda3/envs/trellis2/bin/python"
CKPT_DIR="$ROOT/results/multiview_1186_ss_flow_lora_64_views/ckpts"
DEC="$ROOT/checkpoints/ckpts/shape_dec_next_dc_f16c32_fp16.pt"
INPUT="$ROOT/data_test"
BASE_OUT="$ROOT/results_infer/data_test_stage1_lora_stage2_pretrained_1024_20k_29k"
mkdir -p "$BASE_OUT"

GPUS="4 5 6 7"
set -- $GPUS
GPUS=($GPUS)

run_ckpt() {
    local gpu="$1" n="$2" i="$3" total="$4"
    step=$(printf '%07d' "$n")
    ckpt="$CKPT_DIR/denoiser_ema0.9999_step${step}.pt"
    out="$BASE_OUT/step$(printf '%06d' "$n")"
    log="$BASE_OUT/run.log"
    if [[ ! -s "$ckpt" ]]; then
        echo "[$(date -Is)] ERROR missing checkpoint: $ckpt" >> "$log"
        return 1
    fi
    mkdir -p "$out"
    count=$(find "$out" -maxdepth 1 -type f -name '*.glb' -size +0c | wc -l)
    if [[ "$count" -eq 8 ]]; then
        echo "[$(date -Is)] [gpu$gpu] skip step=$n (8 GLBs present)" >> "$log"
        return 0
    fi
    echo "[$(date -Is)] [gpu$gpu] start step=$n ($i/$total)" >> "$log"
    CUDA_VISIBLE_DEVICES="$gpu" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      "$PY" -u "$ROOT/scripts/infer_finetuned.py" \
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
    echo "[$(date -Is)] [gpu$gpu] finish step=$n glbs=$count" >> "$log"
    if [[ "$count" -ne 8 ]]; then
        echo "[$(date -Is)] [gpu$gpu] ERROR incomplete step=$n, see $out/infer.log" >> "$log"
        return 1
    fi
    return 0
}

cd "$ROOT"
STEPS=( $(seq 20000 1000 29000) )
gpu_i=0
pids=()
for i in "${!STEPS[@]}"; do
    gpu="${GPUS[$gpu_i]}"
    gpu_i=$(( (gpu_i + 1) % 4 ))
    run_ckpt "$gpu" "${STEPS[$i]}" "$((i+1))" "${#STEPS[@]}" &
    pids+=($!)
done

fail=0
for p in "${pids[@]}"; do
    wait "$p" || fail=1
done
if [[ "$fail" -eq 0 ]]; then
    echo "[$(date -Is)] all steps finished successfully" >> "$BASE_OUT/run.log"
else
    echo "[$(date -Is)] some steps FAILED" >> "$BASE_OUT/run.log"
fi

#!/usr/bin/env bash
# ── 中文摘要 ─────────────────────────────────────────────
# 用途：在 side/005_side.png 上扫描 SS LoRA 各步数（1k..12k，每 1k 一档），1024 管线、其余权重保持预训练。
# 用法：bash scripts/inference_history/infer_ss_lora_steps.sh（无参数；路径写死 /data/lzx/TRELLIS.2，GPU 固定 7）。
# 输出：results_infer/side_ss_lora_steps/step<NNNN>/{005_side.glb,infer.log}；已有该 glb 的档位自动跳过。
# 边界：会 mkdir 输出目录并跑 GPU 推理；只读权重，不训练。
set -u
ROOT="/data/lzx/TRELLIS.2"
CKPT_DIR="$ROOT/results/multiview_1186_ss_flow_lora_64/ckpts"
BASE_OUT="$ROOT/results_infer/side_ss_lora_steps"
mkdir -p "$BASE_OUT"

for n in $(seq 1000 1000 12000); do
    step=$(printf '%07d' "$n")
    out="$BASE_OUT/step$(printf '%04d' "$n")"
    mkdir -p "$out"
    if [ -f "$out/005_side.glb" ]; then
        echo "skip step $n"
        continue
    fi
    echo "start step $n"
    CUDA_VISIBLE_DEVICES=7 /home/test/miniconda3/envs/trellis2/bin/python -u \
      "$ROOT/scripts/infer_finetuned.py" \
      --image "$ROOT/side/005_side.png" \
      --output "$out" --pipeline_type 1024 --pretrained_only \
      --sparse_structure_lora "$CKPT_DIR/denoiser_ema0.9999_step${step}.pt" \
      --ss_lora_rank 8 --ss_lora_alpha 8 --no_video \
      > "$out/infer.log" 2>&1
    rc=$?
    echo "step $n exit $rc"
done
echo "all steps finished"

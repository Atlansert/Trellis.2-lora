#!/usr/bin/env bash
# 1536_cascade inference: ss 28k + 512 LoRA 50k + 1024 LoRA 20k
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
PY="/home/test/miniconda3/envs/trellis2/bin/python"
OUT="$ROOT/results_infer/data_test_1536cascade_ss28k_512lora50k_1024lora20k"
export CUDA_VISIBLE_DEVICES=6
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT"
cd "$ROOT"
"$PY" -u scripts/infer_finetuned.py \
  --input_dir "$ROOT/data_test" \
  --output "$OUT" \
  --pipeline_type 1536_cascade \
  --vae_decoder "$ROOT/checkpoints/ckpts/shape_dec_next_dc_f16c32_fp16.pt" \
  --sparse_structure_lora "$ROOT/results/multiview_1186_ss_flow_lora_64_views/ckpts/denoiser_ema0.9999_step0028000.pt" \
  --ss_lora_rank 8 --ss_lora_alpha 8 \
  --flow_512_lora "$ROOT/results/multiview_1186_flow_512_lora/ckpts/denoiser_ema0.9999_step0050000.pt" \
  --flow_1024_lora "$ROOT/results/multiview_1186_flow_1024_lora_v2/ckpts/denoiser_ema0.9999_step0020000.pt" \
  --shape_lora_rank 8 --shape_lora_alpha 8 \
  --no_video >> "$OUT/infer.log" 2>&1
echo "EXIT=$? 输出: $OUT" >> "$OUT/infer.log"

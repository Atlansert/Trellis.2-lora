#!/usr/bin/env bash
# Stage 2 (1024 shape flow) LoRA — RESUME from latest checkpoint (step 20000)
# to max_steps=200,000 on GPU 4. Same recipe as run_stage2_lora_v2.sh.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
PY="/home/test/miniconda3/envs/trellis2/bin/python"
export CUDA_VISIBLE_DEVICES=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT="$ROOT/results/multiview_1186_flow_1024_lora_v2"
DATA='{"train_data":{"base":"./training_data/multiview_1186","shape_latent":"./training_data/multiview_1186/shape_latents/shape_enc_next_dc_f16c32_fp16_1024","render_cond":"./training_data/multiview_1186/renders_cond"}}'
cd "$ROOT"
exec "$PY" train.py \
  --config configs/gen/shape_flow_lora_1024_v2_train.json \
  --output_dir "$OUT" \
  --load_dir "$OUT" \
  --ckpt latest \
  --num_gpus 1 \
  --data_dir "$DATA" \
  --auto_retry 3
#!/usr/bin/env bash
# ── 中文摘要 ─────────────────────────────────────────────
# 用途：Stage1（SS）LoRA 续训：从 latest checkpoint 续到 1,000,000 步（固定用 GPU 7）。
# 用法：bash scripts/run_ss_lora_64_resume_1000k.sh（无参数）；需 conda 环境 trellis2 的 python。
# 输出：results/multiview_1186_ss_flow_lora_64_views/（同目录续写，log.txt 追加）。
# 边界：长跑 GPU 训练任务；--load_dir 与 --output_dir 同一个目录，会继续在该目录里存 checkpoint。
# Stage 1 (sparse structure) LoRA — RESUME from latest checkpoint (step 390000)
# to max_steps=1,000,000 on GPU 7. Same dataset/config family as the existing
# multiview_1186_ss_flow_lora_64_views run (ss_latents_views_mesh_t1).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
PY="/home/test/miniconda3/envs/trellis2/bin/python"
export CUDA_VISIBLE_DEVICES=7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT="$ROOT/results/multiview_1186_ss_flow_lora_64_views"
DATA='{"train_data":{"base":"./training_data/multiview_1186","ss_view_latent":"./training_data/multiview_1186/ss_latents_views_mesh_t1","render_cond":"./training_data/multiview_1186/renders_cond"}}'
cd "$ROOT"
exec "$PY" train.py \
  --config configs/gen/ss_flow_lora_64_views_train_to_1000k.json \
  --output_dir "$OUT" \
  --load_dir "$OUT" \
  --ckpt latest \
  --num_gpus 1 \
  --data_dir "$DATA" \
  --auto_retry 3
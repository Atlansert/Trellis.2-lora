#!/usr/bin/env bash
# ── 中文摘要 ─────────────────────────────────────────────
# 用途：严格 T1 管线串跑：等指定 PID 的网格体素化结束 → 校验体素 14184 个 → 编码逐视角 latent 并校验 14184 个 →
#       在 GPU 7 启动 SS LoRA 训练（config ss_flow_lora_64_views_train）。
# 用法：bash scripts/training_history/run_strict_t1_lora_pipeline.sh <prep_pid>（需在仓库根下执行）。
# 输出：results/strict_t1_pipeline_status.log、results/strict_t1_encode.log；训练写 results/multiview_1186_ss_flow_lora_64_views/。
# 边界：会写上述日志并启动长跑训练；体素数不等于 14184 退 2，latent 数不等于 14184 退 3。
set -euo pipefail
ROOT="training_data/multiview_1186"
VOX="ss_view_voxels_mesh_t1"
LAT="ss_latents_views_mesh_t1"
PREP_PID="${1:?prep pid required}"
STATUS_LOG="results/strict_t1_pipeline_status.log"
mkdir -p results
echo "[$(date -Is)] waiting for mesh voxelization pid=$PREP_PID" >> "$STATUS_LOG"
while kill -0 "$PREP_PID" 2>/dev/null; do
  count=$(find "$ROOT/$VOX" -name '*.npz' 2>/dev/null | wc -l)
  echo "[$(date -Is)] voxel_files=$count/14184" >> "$STATUS_LOG"
  sleep 60
done
echo "[$(date -Is)] mesh voxelization process exited" >> "$STATUS_LOG"
COUNT=$(find "$ROOT/$VOX" -name '*.npz' | wc -l)
if [ "$COUNT" -ne 14184 ]; then echo "[$(date -Is)] ERROR incomplete voxel files=$COUNT" >> "$STATUS_LOG"; exit 2; fi
echo "[$(date -Is)] voxelization complete; starting latent encoding" >> "$STATUS_LOG"
/home/test/miniconda3/envs/trellis2/bin/python data_toolkit/encode_ss_view_latents.py --root "$ROOT" --voxel_name "$VOX" --output_name "$LAT" --batch_size 32 > results/strict_t1_encode.log 2>&1
COUNT2=$(find "$ROOT/$LAT" -name '*.npz' | wc -l)
if [ "$COUNT2" -ne 14184 ]; then echo "[$(date -Is)] ERROR incomplete latent files=$COUNT2" >> "$STATUS_LOG"; exit 3; fi
echo "[$(date -Is)] latent encoding complete; starting training on GPU7" >> "$STATUS_LOG"
export CUDA_VISIBLE_DEVICES=7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec /home/test/miniconda3/envs/trellis2/bin/python -u train.py --config configs/gen/ss_flow_lora_64_views_train.json --output_dir results/multiview_1186_ss_flow_lora_64_views --ckpt none --num_gpus 1 --auto_retry 0 --data_dir '{"train_data":{"base":"./training_data/multiview_1186","ss_view_latent":"./training_data/multiview_1186/ss_latents_views_mesh_t1","render_cond":"./training_data/multiview_1186/renders_cond"}}'

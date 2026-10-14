#!/usr/bin/env bash
# ── 中文摘要 ─────────────────────────────────────────────
# 用途：只读监控循环：每 60 秒打印体素/latent npz 计数与训练日志的最后一行，直到数据齐了且相关进程都退出。
# 用法：bash scripts/training_history/monitor_strict_t1_pipeline.sh（无参数，需在仓库根下执行）。
# 输出：只往 stdout 打印（不写文件）；计数目标是各 14184 个，达成且无相关进程时才正常退出。
# 边界：纯只读（find/tail/pgrep），不训练、不改任何数据。
set -u
ROOT="training_data/multiview_1186"
while true; do
  now=$(date -Is)
  prep=$(find "$ROOT/ss_view_voxels_mesh_t1" -name '*.npz' 2>/dev/null | wc -l)
  lat=$(find "$ROOT/ss_latents_views_mesh_t1" -name '*.npz' 2>/dev/null | wc -l)
  train=$(tail -n 1 results/multiview_1186_ss_flow_lora_64_views/log.txt 2>/dev/null || echo 'not started')
  printf '[%s] voxel=%s/14184 latent=%s/14184\n%s\n' "$now" "$prep" "$lat" "$train"
  if ! pgrep -f 'generate_ss_view_voxels_mesh_t1|run_strict_t1_lora_pipeline|encode_ss_view_latents|train.py.*multiview_1186_ss_flow_lora_64_views' >/dev/null; then
    [ "$prep" -eq 14184 ] && [ "$lat" -eq 14184 ] && break
  fi
  sleep 60
done

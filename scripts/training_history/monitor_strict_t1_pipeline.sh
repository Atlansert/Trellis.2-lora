#!/usr/bin/env bash
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

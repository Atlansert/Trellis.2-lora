#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="/home/test/miniconda3/envs/trellis2/bin/python"
export CUDA_VISIBLE_DEVICES=7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Keep this TRELLIS.1-style per-view run separate from the older object-level
# experiment, whose checkpoints used a different dataset class.
OUT="$ROOT/results/multiview_1186_ss_flow_lora_64_trellis1_views"
DATA='{"train_data":{"base":"./training_data/multiview_1186","ss_view_latent":"./training_data/multiview_1186/ss_latents_views_v2","render_cond":"./training_data/multiview_1186/renders_cond"}}'
cd "$ROOT"

# TRELLIS.1-style supervision requires one latent per rendered view.
DATA_JSON="$DATA" "$PY" - <<'PY'
import json, os, pathlib
cfg = json.loads(os.environ['DATA_JSON'])
root = pathlib.Path(cfg['train_data']['ss_view_latent'])
render = pathlib.Path(cfg['train_data']['render_cond'])
objects = [p for p in root.iterdir() if p.is_dir()]
pairs = sum(1 for obj in objects for _ in obj.glob('*.npz'))
if len(objects) != 1182 or pairs != 1182 * 24:
    raise SystemExit(f'per-view data mismatch: objects={len(objects)} pairs={pairs}, expected 1182/28368')
for obj in objects:
    tf = render / obj.name / 'transforms.json'
    if not tf.is_file():
        raise SystemExit(f'missing transforms.json for {obj.name}')
    frames = json.loads(tf.read_text())['frames']
    if len(frames) < 24:
        raise SystemExit(f'not enough rendered views for {obj.name}: {len(frames)}')
    for i in range(24):
        if not (render / obj.name / frames[i]['file_path']).is_file():
            raise SystemExit(f'missing condition image for {obj.name} view {i}')
print(f'validated TRELLIS.1-style per-view data: {len(objects)} objects, {pairs} pairs')
PY

exec "$PY" train.py \
  --config configs/gen/ss_flow_lora_64_train.json \
  --output_dir "$OUT" \
  --ckpt none \
  --num_gpus 1 \
  --data_dir "$DATA" \
  --auto_retry 3

# LoRA project status

This repository contains the TRELLIS.2 LoRA source, training configurations and regression tests. Dataset assets, pretrained weights, local checkpoints and experiment outputs are deliberately excluded from Git.

## Training and inference

- SS, shape and texture flow configurations enable attention LoRA with rank 8 and alpha 8. The image conditioning in the pool1302 configurations uses DINOv3.
- The data pool contains 1302 objects (1146 training, 156 holdout). Texture eligibility is a separate subset.
- Current geometry workflow keeps the original SS33k checkpoint fixed. Further adaptation of that SS checkpoint under the newer configuration was rejected by the user after visual inspection. Shape/detail work is the next focus; this publication does not launch training.
- Every image in `data_test/` or `data_for_test/` has NO paired 3D asset. Matching numeric labels to training objects is invalid. No GT-relative width/height error, Chamfer, or quality ranking may be computed against an unrelated same-numbered training mesh. Export GLB files for the user's visual review instead.
- Training-render images may only be compared to their own verified source geometry.
- Canonical raw axes for this dataset are depth index 0, length index 1, up index 2. GLB export uses `(x, z, -y)`. Never infer asset correspondence from filenames.

## Entrypoints

- `train.py`: configuration-driven training and resume/checkpoint contracts.
- `scripts/infer_finetuned.py`: explicit SS/shape/texture checkpoint selection and GLB export; use `--help` for flags. `--fast_glb` produces untextured geometry for review.
- `scripts/run_shape_lora_pool1302_v2.sh` and `scripts/run_tex_flow_lora_pool1302.sh`: local training examples. They contain machine-specific Python paths; adapt those paths and supply your own data before running.
- `configs/`: upstream model templates and project experiment configurations. Historic SS configurations are retained for reproducibility, not as recommendations to restart rejected experiments.
- `tests/`: meaningful regression tests for LoRA injection, coordinate transforms, loading, inference and checkpoint contracts.
- `o-voxel/`: native dependency source. Initialize its Eigen submodule before building as required by upstream.

Install GPU dependencies using the upstream installation instructions and `setup.sh`; supply the pretrained model files and DINOv3 locally. No models or datasets are distributed here.

## Cleanup, 2026-10-18

Removed obsolete pilot checkpoint files from the short cohort/quality experiments, superseded root summaries and regenerable Python bytecode. Kept formal results, training data, old SS33k, shape/texture checkpoints, GLBs, regression tests and provenance reports. The exact local deletion inventory is stored in `maintenance/cleanup_20261018/deletions.json` and is not published.

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

## Completeness audit (2026-10-19)

Core source/config/test directories were compared against the working project. The initial code publication omitted demo assets and local model metadata. The follow-up adds all upstream-tracked demo assets, HDRI license notices, DINOv3 metadata/license/model card, local RMBG Python definitions and model card, and checkpoint architecture/pipeline JSON files. Local pipeline metadata uses project-relative `./dinov3` and `./rmbg` paths instead of author-specific absolute paths.

### External prerequisites

This is a source distribution, not a bundled trained model. You still need:

1. A compatible NVIDIA CUDA/PyTorch environment and the native extensions installed by `setup.sh`. Initialize Eigen using `git submodule update --init --recursive` before building O-Voxel. Install a Transformers release exposing `DINOv3ViTModel` (the working environment used 5.14.1); the older generic minimum in upstream prose is insufficient for DINOv3.
2. TRELLIS.2 pretrained component weights under `checkpoints/ckpts/`, matching the supplied architecture JSONs. The SS decoder reference points to the separate `microsoft/TRELLIS-image-large` model repository and may need network/cache access.
3. DINOv3 ViT-L/16 weights matching `dinov3/config.json` under `dinov3/`. Obtain them from the model provider under its license; the local model card records the model family. No DINOv3 weights are included.
4. Background-removal weights matching the supplied `rmbg/` metadata. The local model card identifies BRIA RMBG-2.0 and its separate terms. Do not infer the checkpoint identity from `config.json`'s inherited `_name_or_path` value (`ZhengPeng7/BiRefNet`); use the matching provider artifact. No RMBG weights are included.
5. Your own LoRA checkpoints and training dataset/latents for project fine-tuning. Neither the private train pool nor the unpaired test photographs are published. Historical launcher scripts contain local interpreter paths; use your own interpreter or invoke `train.py` with the referenced configuration directly.

The unit tests validate CPU contracts and source packaging. They do not prove a fresh GPU environment can train or run inference without preparing those external dependencies.

The USDZ preprocessing path also includes `data_toolkit/blender_script/io_scene_usdz.zip`, the local Blender importer source archive required by the mesh/PBR/render scripts. This archive is a code dependency, not an experiment artifact.

The O-Voxel PEP 517 build declaration (`o-voxel/pyproject.toml`) and upstream overview image are included. The upstream GitHub CodeQL workflow is not part of this source mirror; this does not affect local builds, tests or inference.

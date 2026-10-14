# Six-asset SS evaluation preparation

Only preparation and CPU validation. No GPU/model creation, evaluation sampling or training performed by this subtask.

Remote directory: /data/lzx/TRELLIS.2/diagnostics/data_rebuild_batch_v1/evaluation_prepare.

## Scope and outputs

Compare official SS base (before any adapter injection) to committed live step50. Six exact cached RGB512 camera0 inputs: four train / two internal heldout with independently encoded target occupancy; external data_test001–004 have no matching asset GT and receive only necessary axis/count/cross-section diagnostics. Total20 SS samplings, no shape or texture sampling, no GLB export. Exact depthX/lengthY/upZ frame, no alignment/axis correction.

Every tag resets seed42 and follows get_cond512 -> get_cond1024 -> get_ss_cond; condition/RNG/noise SHA required identical across base/live for each input. Multiple injection is avoided: one inject150 after ten official samplings, strict rank8/alpha8 full state load/eval. EMA50 is not primary because initial lag; no warmup/rate change.

Output metrics contain per-asset IoU/precision/recall, count, spans, covariance/PCA, centroid, up/down mass proxy; train and internal holdout mean AND minimum, never cherry-pick maximum. External outputs never get IoU fidelity claims. Fixed coordinate SS projections and coords NPZ retained. Failure partial evidence retained, output overwrite refused, COMPLETE only success.

## Identity guards

Before model imports, validate actual render gate, six target hashes, pilot manifest dependencies and encoded COMPLETE; exact train/heldout partition, camera0, RGB512, finite FP32SSmean and integer64 support. Actual checkpoint complete group fully hash checked; exact live/EMA/misc topology, runtime contract manifest hash / phase50 / train scope / holdout scope / base record / seed / 4669440 parameters, and actual approval artifact SHA required. No inference of model identity from filename only.

Execution then scans live checkpoint tensors with mmap CPU against official safetensors:300 adapterFP32,640frozen exact agreement, nonfinite0. Actual config/weights/input/output hashes saved.

## CLI

Input-only CPU check:

```text
CUDA_VISIBLE_DEVICES='' /home/test/miniconda3/envs/trellis2/bin/python evaluation_prepare/evaluate_six_ss.py --render-gate batch_render_validation.json --encoded-manifest encoded_targets_v1/manifest.json --pilot-manifest short_training_prepare/contract_v1/manifest.json --output NEW --validate-inputs-only
```

Actual checkpoint CPU check (same command plus):

```text
--checkpoint-dir ss_short_train4_holdout2_step50/ckpts --approval approval_four_train_two_holdout_50.json --step50
```

Use `--step 50` (separate tokens); documentation abbreviation step50 above is descriptive. Only parent may add `--execute` and use CUDA_VISIBLE_DEVICES=6 after completing training and ensuring device idle. Managed600second outer supervisor required; estimated20samplings~60sec+pipeline load~90sec, not measured runtime.

## Tests

CPU tests cover exact no-alignment IoU, correct axis ordering, integer/grid/finite rejection, missing GT external semantics, nonzero batch rejection, overwrite and hash/path refusal, all-object mean/min statistics and incomplete comparisons. Actual six-input CPU contract checked without models. Training checkpoint CPU check will use parent's actual committed step50 only; no fake approval or fabricated checkpoint.

No modifications to production source, dataset, weights, old artifacts or nju. Render neutral-light domain gap and limited four/two cohort remain report boundaries; a better SS proxy does not itself certify formal geometry or production model quality.

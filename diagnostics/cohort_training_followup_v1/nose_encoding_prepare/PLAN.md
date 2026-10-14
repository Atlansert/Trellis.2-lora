# Nose canonical encoding preparation only

No GPU/model/training executed. Actual CPU validation passed for all6, 2 proper rotations and4 exact identity reuse references. 9CPU tests and compilation passed.

CLI: conda python -u encode_nose_targets.py --candidate-manifest ABS/nose_canonical_candidate/manifest.json --stage validate

Later parent-only execution: same CLI plus --output ABS/diagnostics/cohort_training_followup_v1/nose_target_encoded_v1 --stage encode|decode|ss|finalize --execute. Shape encode/decode require CUDA_VISIBLE_DEVICES=6, SS/finalize require empty visibility. Supervisor budget each900seconds, no automatic retry. Two assets expected roughly prior two-asset60seconds but not measured here.

Only changed1208train and1f7cheldout truly rebuild1024dualgrid/shape posterior then decoder andsharedshape64support→SS posterior/decoder. Otherfour identity geometries use hash-verified references to previously actual encoded targets, no repeated models or latent permutations. No fabricated renderCOMPLETE: candidate camera-to-canonical transforms explain unchanged image relation.

Newstage files committed with input/output andmodel dependency hashes; resume skips completeverified only. Partial targets not overwritten, unrelated output rejected. Final manifest references4 unchanged and2 new targets, unchangedcondition hashes, declaredcanonical targetnose-Y/height+Z, trainingfalse.

Evaluate direct IoU/geometry in new target frame only; don't compare oldmixed-frame GT as if unchanged anddon't rotate predictions for acceptance. Conditions exactly unchanged. Sourceanchor choice not derivedfromheldoutpredictions. Actualshape/SS encoding andsemantic roundtrip checks remain required. No production approval or PBR encoding.

One construction derive failed substring before writingrunner, fixed without executingmodels; initial9tests1test-string error corrected, finalall9passed. All originalsrc/weights/data/conditions retained.

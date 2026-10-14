# Stage2 coordinate gate: FAILED (diagnosis complete, training withheld)

Same548 native-alpha provisional objects: SS lengthdim1 vs shape supervision lengthdim2(507)/dim0(41), zero same-long-axis cases. Same64 support identity IoU median .0339956; best diagnostic signed permutation median1,519 exact1, minimum .9324. Best det-1 reaches optimum for all548; actual unique optimum524. Do not apply these individually fitted transforms to generated meshes or use bestfit as acceptance.

Actual Blender4.5.1 native glTF marker import confirms native(Dx,Dy,Dz)->raw(Dx,-Dz,Dy) det+1. Current SS yaw0=(-Dy,-Dz,-Dx): physical up native+Dy becomes negativeSS0 and remains GLB-x, not GLB+y. Native lengthDz maps correct formalGLBz by accident of partial axis mapping.

Existing sparse upsample test: 4levels/8192 leaves, each child remains in same-axis parent voxel. With submanifold conv, decoder cannot globally rotate arbitrary support into shape training axes. Previous broad decoder-global-axis-compensation hypothesis withdrawn.

Two real asymmetric GLB native vertices match saved dump exactly; raw dump support lengths2/0 become formalGLB length1/0, not expected2. This is groundtruth-data diagnostic, not a generated mesh or shape20k causal-collapse proof. Shape20k actual saved command uses shape_enc_next_dc_f16c32_fp16_1024. Current checkpoint integrity remains sound; correct files do not imply correct frame distribution.

NEW canonical_frame.py defines explicit proper common raw lengthY/upZ for native glTF Y-up inputs, 5 CPU marker/orthogonality/yaw/gridtests pass. This helper is not wired into old encoders/pipeline, no existing PERM or weights changed. OBJ up-axis remains to be calibrated separately. Existing cached means/features cannot simply be transposed to repair learned VAE frames.

Minimal future joint-data pilot: same proper mesh frame for dualgrid/shape/PBR and SS64support; independently verify roof/wheels/nose/up handedness and same support; re-encode respective VAEs into NEW version directories. Fixed shape20k mixed-native-frame compatibility must be measured, not assumed. No training approval or new training config issued.

Evidence: stage2_support_frames.json, support_inheritance_result.json, axis_markers_native.gltf, axis_markers_result.json, stage2_real_asset_axes.{json,png}, alpha_scan/stage2_axis_independent.json. Pipeline/model reliability work may proceed safely despite failed training gate.

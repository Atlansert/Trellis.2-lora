# Common-frame Blender renderer preparation

Prepared only; this task did NOT launch Blender or render pixels. Parent owns first-asset runtime and output review. No training, model/GPU use, old source/data changes, installations or node deletions occurred.

## Files / ready invocation

Remote base: /data/lzx/TRELLIS.2/diagnostics/data_rebuild_batch_v1/renderer_prepare

- render_common_frame.py: standalone bpy renderer requiring reviewed local GLB; no use of old OBJ import dispatch.
- render_common_contract.py: CPU-only source review / math / camera / alpha / overwrite contracts.
- test_render_common_contract.py: 18 CPU tests.

First parent invocation:

```text
CUDA_VISIBLE_DEVICES='' /data/lzx/opt/blender-4.5.1-linux-x64/blender --background --factory-startup --python /data/lzx/TRELLIS.2/diagnostics/data_rebuild_batch_v1/renderer_prepare/render_common_frame.py -- --review /data/lzx/TRELLIS.2/diagnostics/data_rebuild_batch_v1/asset_review/0086bbab37559442.review.json --output /data/lzx/TRELLIS.2/diagnostics/data_rebuild_batch_v1/renders/0086bbab37559442_cpu_v1 --device cpu --resolution 512 --samples 16 --threads 8
```

Parent must supervise the new process group with the per-asset 600-second budget, log/PID/exit/result recording and termination of only the newly created worker. Runtime cost has NOT been measured; no 256 preview is counted as 512 rendering. Defaults are actual 512², 16 Cycles samples, 8 CPU threads, radius2.2, FOV30. Those batch-v1 values cannot silently change.

## Review compatibility and protection

Supports exact parent source review format already written by asset reviewer, without rewriting parent approval hashes:

- schema trellis.asset-render-review.v1, review_passed=true, approved_for_render=true;
- source absolute GLB path and SHA256;
- retain_all_reviewed_meshes=true;
- approved_source_mesh_node_names + source_scene_topology.nodes(index/name/mesh), all native mesh nodes must match actual GLB JSON;
- matrix_row_native_to_raw + selected_yaw_degrees, proper matrix independently checked;
- object_id, review_scope, manual_image_observation.

Alternative standalone tests use reviewed_native_mesh_nodes full records + explicit gltf_y_up/reviewer/review_evidence. Both paths are failclosed; no generic placeholder approval is created here. External URI images/buffers are rejected because source GLB hash would not bind them. Parent's six actual asset reviews passed hash/node/matrix/embedded dependency preflight: 0086,008ae,015de,0390 yaw0;1208,1f7 yaw90.

Actual imported Blender object names are recorded. Optional expected_blender_mesh_object_names/count can be strict when known. Otherwise all imported meshes are kept, without assuming trimesh/native names are identical to Blender names. Hidden imported geometry and negative world determinant require separate review; renderer does not flip faces or remove helpers automatically.

## Frame / pixels / geometry outputs

Blender importer already maps native glTF (x,y,z)->(x,-z,y), determinant+1. Renderer adds ONLY column Rz(-selectedyaw), equivalent to approved row Rz(selectedyaw), to a common root. Then all reviewed mesh geometry is centered/scaled to fit .99999 longest extent. Hierarchy-preserving root transforms are checked against explicit world vertices. Canonical raw geometry is depthX/lengthY/upZ; all eight views only rotate the CAMERA. No per-view target mesh rotation/cyclic old SS permutation occurs.

Camera yaws0,45,90,135,180,225,270,315; view0 camera+X/looks-X/up+Z/screen-right+Y. Initial conditional training should use view0 until opposite-side nose handling is explicitly defined. Eight images do not mean eight independently rotated canonical SS targets.

Cycles outputs transparent RGBA8 PNG with film_transparent=true, direct alpha coverage and bbox inspection; no color-background derivation or largest-component filtering. Controlled white environment and Filmic/exposure0/gamma1 are explicit. Imported lights are disabled and recorded, not deleted. Meshes/materials/UV/textures are retained; imported materials are not rewritten. Historical pixel equality is NOT assumed. Glass/material transparency follows real Blender rendering and may need separate semantic mask review.

canonical_geometry.npz contains evaluated, world-transformed vertices/triangulated faces in common raw frame, not texture features. Geometry hashes/bounds/object matrices/material slot names included. Faces from Blender evaluated mesh may differ tessellation from trimesh; runtime geometry comparison must account for vertex ordering/tessellation rather than require identical file hashes.

Per-view metadata saves actual camera.matrix_world and its inverse, explicit K for square512 horizontal32mm sensor, actual lens/angle, camera right/up/forward vectors and world sample projection audit. Pixel convention is top-left image x-right/y-down with principal point width/2,height/2 (pixel-edge coordinates). Alpha bbox uses saved PNG's top-down coordinates.

Manifest includes source/review/script/helper SHA, Blender version, options, normalization/yaw/root matrices, AABB, object inventory, render color convention, camera geometry, per-PNG SHA/time/alpha. Manifest is atomically replaced after each completed view. Partial failure preserves files and failed manifest; new output directory required for retry. COMPLETE is written only after all eight and source/review final hash check. COMPLETE is render completion, NOT training approval.

## Device safeguards

CPU is default. Explicit cuda requires CUDA_VISIBLE_DEVICES exactly6 or7 and exactly one CUDA device discovered by Cycles. All non-CUDA devices disabled; no fallback to GPU0–3 or automatic multiGPU choice. No GPU runtime was exercised by this preparation.

## Verification and remaining checks

18 CPU tests passed (0.039s), including proper markers/determinants/all-yaw up preservation, row/column rotation equivalence, normalization, camera rigid inverse/projected right/up/origin for all eight cameras, direct alpha empty/opaque rejection, changed source/node whitelist rejection, external dependency rejection, overwrite rejection and CUDA selection restrictions. py_compile passed for renderer/helper/tests without bpy execution.

Remaining: actual Blender4.5 RNA/API behavior, actual source-material import/rendered RGB/alpha, execution time, exported world geometry against previous proper pilot and real pixel projection. First runtime must be reviewed by parent before batch. No render approval expansion or production training approval is inferred from math/tests/preflight.

# Rebuilt six-asset SS live50 formal GLB validation v2 (all four completed)

CPU-only scope: 4 new formal GLBs against 4 old33k baselines. Every scene node instance applies its complete world transform on copies; no model/GPU, training, source geometry edits or coordinate correction. Source weights/artifacts unchanged.

## Results

|record|runtime provenance|AABB longest|exact surface PCA offZ deg|z/y|x/y|degenerate faces|
|---|---|---|---:|---:|---:|---:|
|old33k_001|provenance_unverified|z|0.446634|8.573730|0.951728|0|
|old33k_002|provenance_unverified|z|0.893576|6.372133|0.697079|0|
|old33k_003|provenance_unverified|z|1.413388|6.051773|0.995261|3|
|old33k_004|provenance_unverified|z|0.288766|8.838790|0.913781|0|
|rebuilt_ss_live50_001|verified|z|0.170272|6.178956|1.432265|0|
|rebuilt_ss_live50_002|verified|x|88.771247|1.280779|5.403788|0|
|rebuilt_ss_live50_003|verified|z|0.971704|5.155537|1.096904|0|
|rebuilt_ss_live50_004|verified|z|0.349583|8.691792|1.120515|0|

**Overall acceptance FAIL. New002 is longX, not required z; exact PCA offZ88.77deg confirms a necessary orientation failure.** Reference10deg remains a diagnostic threshold, not a replacement production rule. New001/003/004 align length to z but do not thereby pass geometry fidelity. All geometry_not_worse statuses remain unverified pending parent review; no ratio, covariance, silhouette or smoothness automatically passes quality. Short50 weights remain diagnostic only, no recommendation to expand/continue training.

New all4 provenance is verified from trellis.inference-run.v2: actual input/output hash and size, successful image/run, seed42, pipeline1024, actual false canonicalize/swap flags, formal_remesh_textured export, strict loaded model roles and recorded base/override source metadata. Checkpoint payload records use live path/size and recorded SHA, no repeated tens-of-GB rehash here. Old four provenance remains unverified: historical declarations/manual geometry reviews do not replace missing full runtime evidence.

## Complete-triangle fixed-frame visual review

Image fixed_side_top_all_triangles.png was downloaded and actually read_image inspected. Columns: input (no known GT identity), old sideZY, new sideZY, old topZX, new topZX. All triangles included, same350px/worldunit scale, AABB-centering translation only, no rotation/ICP/best-view or scale fitting. New002 is end-on in fixed side and long vertically in top; do not rotate it to hide failure. New001 is markedly wider transversely; new003 side/back-nose shape differs and height grows; new004 has smaller changes but still no fidelity approval. Grey silhouettes ignore texture and do not resolve 3D fine detail; they are not photoreal renders or signed semantic keypoints.

Missing003/004 v1 snapshot remains immutable in ../v1_first_two; final v2 newly reads all8 completed files. Initial v1 plotting attempt observed new003 appearing after snapshot and wrongly indexed absent metrics; diagnosed report-v-filesystem race, changed missing-panel condition to immutable reportstatus, reran CPU. No outputs or original meshes deleted.

Existing fixed-frame acceptance tests15 passed (../fixed_frame_tool_tests.log, exit0). Final8case CPUreport+plot107.30sec combined, plotting90.72sec. No persistent worker, GPU process, network generation or production source edits started by this evaluation.

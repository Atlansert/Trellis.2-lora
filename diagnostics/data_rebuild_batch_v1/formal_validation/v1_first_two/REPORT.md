# Six-asset rebuilt SS live50: formal GLB CPU evaluation v1

Scope: old33k four images versus rebuilt SS live50 four REQUIRED images. This immutable first snapshot contains new001/002 only; new003/004 were absent when report was evaluated. They remain missing even if produced afterward. New v2 will be generated only after parent notifies all four complete. No GPU/model inference or training in this task.

Runtime provenance: new001/002 trellis.inference-run.v2 verified against actual input/output SHA, seed42, pipeline1024, actual ss_canonicalize=false/ss_swap_yz=false, formal_remesh_textured and loaded model records. Old baseline four files remain provenance_unverified, not converted into certified runtime records.

|image|old PCA offZ deg|new PCA offZ deg|new AABB long axis|new z/y|new x/y|
|---|---:|---:|---|---:|---:|
|001|.446634|.170272|z|6.178956|1.432265|
|002|.893576|88.771247|x|1.280779|5.403788|
|003|1.413388|missing|missing|missing|missing|
|004|.288766|missing|missing|missing|missing|

New002 conclusively fails the original long-axis-z requirement: it is longX. No canonicalization, rotation, PCA alignment, ICP or best-view search was applied. Reference10degrees is diagnostic only, not an invented production threshold. New001 has axis alignment but greater transverse width and different shape; AABB ratio/PCA does not certify fidelity or improvement over baseline. All geometry_not_worse statuses remain unverified pending explicit parent review; total acceptance fails.

Full-triangle CPU union silhouette image uses every face and all scene-instance world transforms, fixed ZY side and ZX top. Common scale350px/unit; AABB-centering is translation-only for display. Grey silhouette is not textured rendering or photoreal mesh quality. It confirms new002 appears end-on in fixedZY and long vertically in fixedZX; geometry was never rotated to hide this.

Evaluation12.56sec; image68.08sec. First image attempt failed because new003 appeared after the immutable missing report and code checked filesystem rather than reportstatus; diagnosed KeyError metrics, corrected missing-panel logic, reran CPU only. No partial image existed/overwritten, report snapshot not changed.

Existing fixed-frame tool15CPUtests rerun pass (see ../fixed_frame_tool_tests.log). Historical003 degeneratefaces3 at relativearea tolerance; others including new001002 zero; this count alone is not fidelity verdict. New short-trained50 checkpoint is diagnostic only, no recommendation to continue training or production approval. Parent continues003004 independently; this task never duplicates GPU inference.

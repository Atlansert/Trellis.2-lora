# Canonical nose target hypothesis: geometry-only candidate

No GPU/models/reencoding/training. Original geometry, conditions, features, SS means, checkpoints and formal outputs unchanged.

Fixed source-only semantic decision: 0086/008ae/015de/0390 nose -rawY;1208/1f7c nose +rawY. Existing source0/source180 image contact actually read and source taper candidates agree. Labels remain visual review, not measured keypoints. No heldout prediction/IoU used to choose transforms.

New target definition: canonical vehicle nose -rawY, up+rawZ, lengthY; RGB inputs stay exactly unchanged. Four identity source references; only1208(train) and1f7c(heldout) new geometry NPZ rotate proper Z yaw180: (-x,-y,z), det+1. Faces/winding unchanged, normal transform checked, height and extents preserved; this is a NEW supervised target pose convention, not posthoc model-output correction. Camera-to-canonical matrices recorded explicitly. In transformed cases same source image camera becomes -rawX, so unchanged image screen direction does not match canonical+X projection; that is the intended pose-invariant mapping, not evidence of bad pairing when stated explicitly. The model is not given camera pose tokens, and whether it can learn this target definition is a future hypothesis, not proven improvement.

Control: same original official initialization, rank/alpha/lr/dtype, fourtrain/twoheldout split, view000 RGB512 caches and sampling evaluation. Change target convention only. Do not combine with 100/200step duration change in one causal claim. Parent duration200 experiment must finish first; if considering canonical nose afterward, new hashbound target approval and fresh same-duration baseline are required.

Required next gates: parent semantic anchor review; true shape dualgrid+encoder recomputation from rotated geometry; SS mean encoding from corresponding newshape64support; numeric/semantic roundtrip; bound experiment approval. Do NOT rotate old latent means or coordinate-only shape features. Current new_shape_latent/new_ss_mean all null and training_approved=false. Evaluation must compare to declared newcanonical targets directly; no best angle/rotation acceptance. External formal GLB must remain uncorrected.

Actual CPU construction8.15secexit0. Eight CPU tests passed, including allactual source andconditionSHA, selectedscope, properdet/up/involution, normals/winding/extents, explicit camera transform, no ready latents,completionhash. Plot is fixed vertex projection diagnostic (not fulltriangle silhouette or surface quality); actual read_image shows first4unchanged and last2noseleft after proper180 with original conditions unchanged. No source files modified and no worker remains.

Outputs manifest.json/COMPLETE.json, two geometry_candidate.npz, identity refs and source_old_candidate_contact.png. Sourceoriginal files remain under data_rebuild_batch_v1; GPU6 model work belongs to parent/other agent and was not touched.

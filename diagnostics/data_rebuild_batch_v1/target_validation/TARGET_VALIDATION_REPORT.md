# Independent six-asset target validation

Data validation passed; scoped short-training data gate passed ONLY for the six reviewed assets, condition view000. No production approval, no all48-view approval, no all543-cohort approval. Parent must bind source/render semantic reviews and bounded training plan separately.

## Actual execution
CPU-only CUDA_VISIBLE_DEVICES empty; 23.795 seconds, foreground tool exit0. No GPU training, model updates, dataset rewrite, PERM changes or source edits. New validation outputs only.

- Batch and encoded COMPLETE/manifest binding checked. Every stage commit equals final manifest; all stage artifact hashes and source geometry/render manifest/source GLB/review hashes match. Four current encoder/decoder JSON and safetensors hashes independently recomputed.
- Six shape arrays: coords uint8 unique grid64; feats and mean FP32 finite Nx32, exactly equal. Dualgrid integer grid1024 support coarsened by16 equals shape support; shared SS support equals shape support. This equality is constructed, not independent geometry fidelity.
- Six SS means FP32 finite 8x16x16x16. Actual CPU float32 decoder called independently on all six means; every occupancy IoU=1.0 and coordinate set equals saved decoded output.
- Every decoded shape geometry finite, faces integer and in bounds. Exact triangle-integrated surface covariance independently recomputed from source and decoded mesh; matches committed stage values. No best rotation/scale/ICP.
- Maximum relative extent discrepancy is0.09054% (1208 height); all under0.5% reference numerical screen, not universal semantic quality criterion.
- Previously executed512 points/side to exact-triangle surface distance results reviewed through hash-bound commits: symmetric means4.3605e-5..6.3109e-5, maximum individual sampled distance.0014057 (1f7c). Proximity itself was not repeated in this independent run; sampled distances do not certify all local detail.

|asset|shape tokens|independent SS IoU|max extent difference percent|source/decode lengthY angle|
|---|---:|---:|---:|---|
|0086bbab37559442|2033|1.0|0.02328|0.74903/0.74130|
|008ae427b2606cc8|2848|1.0|0.02352|1.65155/1.65216|
|015de7a65b71fbc2|2476|1.0|0.04080|0.81858/0.81672|
|0390f705ac242a39|2882|1.0|0.02391|0.46743/0.44819|
|1208e4cc0c30448a|2218|1.0|0.09054|0.33828/0.33158|
|1f7cc1ae1cec73f5|2167|1.0|0.05076|0.64747/0.65242|

## Limits
Numerical target contract only, not trained-model geometry acceptance. Semantic roof/wheel/nose labels are not automatically propagated through VAE. Source scene review and transparent-render/camera reviews are separately completed by parent; this run did not rerender48 views or repeat camera raster audits. Texture/PBR not encoded. Unknown realworld semantics beyond reviewed source remain outside scope.

## Files
validation.json full independent measurements; COMPLETE.json binds validation SHA; scoped_data_gate.json states exact six/view0 data scope. validate_targets_independent.py at batch root is the executable CPU reproducibility script. Original/new training target files remain unchanged.

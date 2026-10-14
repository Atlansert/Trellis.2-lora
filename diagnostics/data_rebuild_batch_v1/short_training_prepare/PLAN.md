# Rebuilt six-asset SS short training preparation

Preparation completed; no GPU model construction, no training, no persistent approval issued by this subagent.

## Exact cohort

Train4: 0086bbab37559442,008ae427b2606cc8,015de7a65b71fbc2,1208e4cc0c30448a.
Internal holdout2: 0390f705ac242a39,1f7cc1ae1cec73f5. No holdout rows sampled by the training Dataset. Initial condition view000 only; remaining seven camera views are geometry audit images, not paired blindly to shared canonical targets.

## Data contract

Targets are actual encoded_targets_v1/{object}/ss_mean.npz, fp32 finite8x16x16x16, proper raw(depth,length,up); no feature/mean axis permutation. Cached conditions reproduce actual production pipeline alpha>204 crop_scale1, alpha premultiplication RGBquantization then actual DINO list-image LANCZOS512 resize. Test executes extracted current pipeline preprocess and verifies pixel equality. CachedRGB512 tensor is the exact DINO512 list-image content.

Manifest binds encoded/source-review/render-batchvalidation hashes, each renderedRGBA/view000 manifest/canonicalgeometry/newSSmean/cachedRGB/source-review file hash and size. Dataset validates dependencies before CUDA and target/image hashes on each item access. Wrong split/extra views/nonfinite or wrong dtype/shape/unsafe path/changed hash is refused. Shape mean remains outside SS trainer; targets include shape-derived support previously validated by independent data gate.

## Recipe

Official SS1.3B base, rank8 alpha8 dropout0, 150 attention layers/300 fp32 adapter tensors only, pretrained frozen. AdamWlr1e-4 wd0/batch4split2/seed42/CFG.1/logitNormal mean1std1/AMPbf16. First phase exactly50 steps, save50, log10, prefetchFalse, skip_snapshotTrue. Dataset visualize accepts dict proxies without pretending geometry monitoring. EMA.9999 not warmed up; live50 primary comparison, EMA only reference.

No automatic100 continuation or resume is implemented. Evaluation should compare original SS and live50 on identical train4, internal2 and external001-004 conditions/seed/noise, with knownGT IoU only where matching rebuilt geometry exists. SS is proxy, not formalGLB acceptance. Do not promote this short weight to production.

## Independent approval

--execute requires parent trellis.six-render-ss-approval.v1 binding manifestSHA/exactsplit/phase50, scoped_data_gate.json hash(size), render_visual_review.json hash(size), TRAINING_DATA_SPLIT.json hash(size), and encodedmanifestSHA. Scoped gate exact6/view0/data_validationpassed/approvedshort/prodfalse plus underlying validationproofSHA checked; visualreview exact6/view0/passed; splitexact4/2/phase50/noauto100. Child created no persistent approval. CPU temporary approval fixture checked real Datasetgetitem4 and was removed.

## Verification and operation

18 CPU tests passed with ResourceWarning errors, py_compile passed. SourceSHA and manifestSHA in PREPARATION_RESULT.json. Default CLI validates CPU and exits. Parent must supervise new workerprocessgroup with900second outer deadline, stdout/err log/PID/exit and onlyCUDA_VISIBLE_DEVICES6. Require32GiB free conservative; one50step live+EMA checkpoint about5.3GB, without modifying prior data/output. fresh_output refuses any existing directory; no implicit retries/cleanup.

Only new diagnostic files were written. No production dataset registration/source edits, GPU0-3 access, NJ changes or oldcheckpoint/config/data changes.

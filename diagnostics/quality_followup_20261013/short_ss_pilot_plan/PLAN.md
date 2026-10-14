# Short SS target-frame pilot: prepared, NOT RUN

## Updated geometry gate

0fab4af3f9a73bdd source contains opaque front/side reference-photo rectangular planes. It is QUARANTINED and this plan cannot approve it. Candidate records are preserved for audit; default training approval is false. Only reviewed0086 can later be approved explicitly. This is then SINGLE-ASSET learnability/overfit diagnostic, not two-object generalization or production improvement. No model constructed, no GPU invocation, no training performed by preparation.

## Single variable

A legacy SS mean at view000, frame label down/length/depth; B newly CPU encoded proper SS mean directly matched to real shape64 support2033, frame depth/length/up. Same accepted object, same cached RGB512 condition, same base1.3B, rank8/alpha8/dropout0, seed42, lr1e-4 AdamWwd0, batch2/split1, CFG dropout.1, logitNormal mean1/std1, AMPbf16 withfp32adapter, EMA.9999. Max100 preferred;200 only separately approved. Save50/100 (or100/200). No opposite-side mixture; no target-feature spatial transposition.

Condition cache is the actual production pipeline preprocessed black RGB from previous successful compatibility run, resized exactly512PIL LANCZOS (same as feature extractor list-image path), loaded as RGB/255 CHW tensor; no second alpha crop. DINO trainer stillbf16 for BOTH arms, unchanged. Scope-one exposes2repeated drawslots because existing trainer uses drop_last=True/batch2; distinctassetcount remains1, draws duplicate sameimage/target and are disclosed, never fake metadata.

## Approval and invocation

run_short_ss_pilot.py DEFAULT validates CPU and exits. --execute additionally requires parent-authored approval schema trellis.short-ss-pilot-approval.v1, bound manifest SHA, approved=true,diagnostic_only=true, objects=[0086...], max_steps100/200, geometry_review_passed and semantic_review_passed plus hash-bound JSON evidence whose passed=true and objectscope covers approved IDs. No TRAINING_APPROVAL is created here. All original dataset contracts remain unmodified. Wrongarm/frame/hash/path/NaN,unapproved or quarantinedscope fail before CUDAmodel construction. Output freshonly.

Conditional command only after parent approval:
CUDA_VISIBLE_DEVICES=4 PYTHONPATH=/data/lzx/TRELLIS.2 /home/test/miniconda3/envs/trellis2/bin/python -u run_short_ss_pilot.py --manifest manifest.json --arm legacy --approval PARENT_APPROVAL.json --output NEW_ARM_A --execute
Then same approved100steps armproper in a second freshoutput. Run sequentially with outer processgroup deadline900seconds each; stop on nonfinite loss/grad, anycontractchange, workerfailure/OOM, boundedtime, or mandatoryfirst50-step evaluation strongly worseproper support/up. Current runner uses existing trainer; per50step external eval/earlystop orchestration is PARENT RESPONSIBILITY, not yet integrated or testedGPU. --execute is NOT authorization to bypass these stop checks.

## Evaluation plan before spending100steps

Evaluate pretrained base, live50,live100, not shortEMA ranking (.9999 half-life6931steps; initialEMAweight100steps99.005%). Conditions train0086 and held-out001/004, seed42, no ss_swap_yz/canonicalize. Sample with12steps/CFG7.5/rescale.7/interval[.6,1]/rescale_t5. Decode support and report properGT supportIoU only ontrainasset, rawlengthaxis1/upaxis2 semanticroof/wheels markers, same-camera profile, generated decodedoccupancyPCAangle/spans, targetlatentMSE separately. Heldout001/004 have no paired3DGT: proxyorientation/profile only, not fabricatedIoU; SSproxy NEVER wholeGLB geometry acceptance. A predictslegacyframe bydesign; do not compare rawAproperIoU and announceBsemanticbetter without acknowledging incompatibletargetdefinition. Learning-fit eachown-target plus common-frame proper compatibility should be reported separately.

At50: verifycontract/finite/FP32adapter/basefrozen and evaluate beforecontinue; abort if new target not learnable or heldoutgrosscollapse. 100steps cannot prove convergence/generalization. No model weight deployed. Localgeometry source0086 GT alreadypassed properSS/shape encode-decode, but batchdraw semantics sourcegeometrygate unreviewedothers remainfalse.

## Budget

Measured actual full SS EMA checkpoint2638351525bytes (~2.46GiB). Live+EMA at2savedsteps across2arms:8model files~21.1GB (~19.7GiB), plus optimizer/misc anddiagnostics; reserve32GiB. Runner checksfree32GiB beforeconstruction/output. WholeCPUclones/multiGBhash increase saveRAM/IO. Two100step trainingcompute rough2–5minutes/arm from prior20krun speed (~.67s/step), model/DINOload~1–2min, synchronous save/eval additional; outer900s/arm is budget not measuredrun guarantee. Parent must monitorGPUmemory/resources beforehand, onlyGPU4–7, keepservices. No automaticretries/restarts by runner.

## Preparation verification

Standalone diagnostic loader uses existingImageConditionedFlowMatchingCFGTrainer without modifying productiondataset/register/config. CPUtests covervalidmatchingcondition,missingapproval,NaNtargetevenrehashed,checksumchange,pathtraversal,wrongframe,scopefalse,quarantinedobject,overwrite. Filesallnew, originaldataweights/NJ/PERM unaffected. Parentapproval remainsmissing.


## Parent-requested safe phase update
Runner now defaults --phase_steps50, i_save50, and exits normally after first committed50-step group. No background continuation to100. --phase_steps100 is permitted only within approved100/200 total but must NOT be used until parent reviews first50. Phase2 resume is deliberately NOT implemented in this preparation; parent can separately implement tested loadstep50 into a freshoutput if needed. Firstphase twoarms writes4full model files ~10.55GB (9.83GiB) plusoptimizer/state;32GiB guard remains conservative. Pilotrun recordsmodelargs/actualbasehash/runner+contracthash/phase vsapprovedtotal.

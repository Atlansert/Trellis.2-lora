# CPU preparation: independent canonical-nose target trial

Prepared only: no GPU model construction, training, or persisted approval.

Fresh official SS1.3B / LoRA rank8 alpha8 / AdamW lr1e-4 / batch4 split2 / seed42 / 100steps / saves50,100. Exact recipe and model args must match original100 run. Four training objects, two heldout; six cached conditions unchanged. Only1208 training and1f7c heldout targets replaced by actual proper Rz180 geometry/VAE encoding. Other four target references remain exact identities. No prediction rotation or best-angle acceptance.

--old-manifest original six pilot manifest --new-target-manifest nose_target_encoded_v1/manifest.json --candidate-manifest nose_canonical_candidate/manifest.json --baseline-run-contract ss_same_cohort_step100/pilot_run_contract.json. Default CPU validation. --execute additionally requires new parent approval and fresh --output; CUDA_VISIBLE_DEVICES=6, 32GiB free; parent external900sec supervisor required.

Approval schema trellis.nose-canonical-target-trial-approval.v1; approved=true diagnostic_only=true production_approved=false max_steps=100, exact4train2holdout, condition_view=0, base_sha256, old/new/candidate/baseline manifest SHA, target_validation path/SHA. Gate must be actual trellis.nose-target-independent-validation.v1 passed/approved_for_scoped_target_trial, unchangedconditions, productionfalse, exactsix objects from rows and encodedmanifestSHA. No approval issued by preparation.

12 CPU tests passed; actual encoded six preflight and temporary parent-approval fixture plus four dataset CPU samples passed. No models loaded. Unit tests cover stage100 vs200/oldapproval/scopeleak/reflection/hash/NaN/dtype/path/output rejection. Source files and old data unchanged. PREPARATION_RESULT.json contains exact hashes.

Evaluate old100 and new100 against the same newcanonical GT with direct metrics, and preserve old-frame metrics separately. Lower error obtained solely by changing GT is not improvement. No 200step/time/rank/lr confound; no production promotion. GPU nondeterminism means full bitwise trajectory equality remains to be checked, not presumed.

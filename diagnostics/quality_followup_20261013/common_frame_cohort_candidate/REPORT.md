# Common-frame cohort metadata candidate

## Independent verification supplement

548 full source GLB SHA256 values all begin with their recorded 16-character object IDs; no source identity mismatch. Source scene node mesh/child indices are valid. All planned matrices were separately checked for orthonormality, determinant +1, and native +Y mapping to raw +Z; selected target views belong to accepted input sets.

Planned yaws over the original 548: 507 at 0°, 41 at 90°. Among 543 provisional candidates: 506 at 0°, 37 at 90°. This preserves selected side conventions, not universal nose sign.

All 548 metadata rows say num_views=0, while every actual transforms frame table has 24 entries. The candidate records both; it relies on validated frame tables, not stale metadata num_views. Historical camera matrices remain placeholders, not real extrinsic proof.

The first and only execution completed in 29.94 seconds (front-end tool 30.18 seconds), exit code 0. No worker/background job was launched, and no process remains. Source file bytes were hashed in four CPU threads; only GLB JSON/header metadata was decoded, not positions, textures, learned latents or meshes.

The cohort is a **next-stage plan**, not trainable data. Old learned SS/shape targets are deliberately not linked as ready targets. Five risk assets stay quarantined; source ownership rights are not independently certified beyond their location in the user's project. All 543 unflagged assets still require source/semantic review and actual shared-frame reencoding/roundtrip.


CPU source JSON/header/hash and accepted-input/frame verification only. No positions, textures or latents decoded; no GPU, reencoding, training, cleanup or original files changed.

{
  "input_admitted_objects": 548,
  "known_risks_quarantined": 5,
  "candidate_unflagged_objects": 543,
  "source_format_counts_all": {
    ".glb": 548
  },
  "candidate_format_counts": {
    ".glb": 543
  },
  "status_counts": {
    "provisional_source_candidate": 543,
    "quarantined": 5
  },
  "source_hashes_computed": 548,
  "accepted_images_verified": 1096,
  "eligible_GLb_provisional": 543,
  "ready_targets": 0,
  "training_approved": false,
  "errors": [],
  "warning": "Unflagged means no recorded geometry proxy flag, not clean geometry/semantics or training readiness; source native Y-up and target nose require review. Old means/features are not linked as new ready targets.",
  "elapsed_seconds": 29.93551808700431
}

All entries ready_target=false; new_shape_latent/new_ss_mean null. Required source/semantic reviews and actual geometry/VAE reencoding remain outstanding. Five known risk assets are quarantined (3 confirmed reference cards,2 uncertain proxies), not automatically deleted or all called corrupt. For GLB only, planned proper native-to-common matrix uses selected accepted-view yaw and preserves that side; no universal nose sign assertion. OBJ orientation cannot default to glTF Y-up. Historical num_views and placeholder camera matrices are retained with explicit limitations.

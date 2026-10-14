# Reference-card suspects and isolated cleanup candidate

CPU only. No original mesh/data/PERM/weights/source changes, no GPU, VAE or training. New outputs: diagnostics/quality_followup_20261013/refcard_cleanup_candidates.

## Two newly reviewed suspects

- f50d6a143bdfe293 / CR400BF0_P: nodes CR400BF01 and CR400BF; geometries CJ5-0506-Side Front.001 and CJ5-0506-Side Front; each exactly2 triangles. Material BLEND, embedded atlas5400x1080, alpha all255. Read contact image confirms white-background front+side train reference photo atlas, not actual modeled vehicle surface. Both nodes share exact pixelSHA bfe5835215cf3f10067c75fcf5dd0fdec6216898e0cc8aeff747725b54a00e37.
- 5d4a5e8e46991642 / CR400AF_01: nodes Side/Front, Plane.001/Plane, each2triangles. BLEND, embedded atlas2560x1440, alphaall255. Contact visibly contains white-background train front/side reference drawing/photo. PixelSHA f4998efcafa8b4bf12d3de994c244c9139317780a59f8a3071525ec5ef1e9e92.

These atlases do NOT match0fab pixelSHA a9a3f943341d4c3593de573214b603b7362cdca29199d7b3d97c72d78b571f07; common modeling pattern, not a claimed exact shared image. Suspects confirmed by scene/node+material+actualimage evidence, not merely BLEND or large triangles. They remain unmodified; review.json retains previsual pending records and review_visual_confirmed.json appends explicit visual results.

## Authorized NEW0fab candidate

Only explicit Fromt/Side referencephoto nodes removed in a scene copy. Unknown node1435/Cube.001 retained, no automatic BLEND removal. Remaining material vehicle instance and cube keep original world transforms, faces, UVs, embedded RGBA texture pixels and alphaMode; exported GLB reloaded and each checked equal (worldvertex atol1e-7). Serialized glTF JSON bitidentity is not promised, tested properties are recorded.

Original sourceSHA 0fab4af3f9a73bddb8dfd5e744cb92dba7efffa82e6d4a5b46d51e31049988b1 remains unchanged.
CandidateSHA 0ec2e64afbea940050622e77a94ea5407f40e8194e383f37a0b5b0dfad88e630.
Filename0fab4af3f9a73bdd_remove_Fromt_Side_candidate.glb in new folder. Original4geometryinstances→candidate2, node1435/material remain. No normalization rescale: native worldbounds recorded before/after.

Full-triangle samecamera silhouette uses identical ORIGINAL normalized scale and common raw+X camera (+rawY screenright/+rawZ screenup), no bestfit. Read image confirms large rectangle removed, tapered nose and wheel silhouette visible again. Unknown cube appears as small front-adjacent block and is intentionally retained. Candidate is NOT automatically semantic-perfect, not encoded, not approved trainingGT. Asset review may later classify cube or other defects.

## Files and execution

reference_atlas_contact.png (three assets actual texture views), cleanup_same_camera.png (original/candidate), review.json/review_visual_confirmed.json, candidate_roundtrip_validation.json, copied per-node textures, newcandidate GLB, reproducible review_cleanup_reference_cards.py. CPU main前台exit0,41.668seconds; roundtripcheck2.702seconds. No background worker.

Parent should exclude the two confirmed new reference-card assets from trusted-original-data assumptions pending scoped cleanup;0fab newcandidate can be reviewed, not silently replace original. No changes made to newotherasset candidates because parent only authorized0fab whitelist export.

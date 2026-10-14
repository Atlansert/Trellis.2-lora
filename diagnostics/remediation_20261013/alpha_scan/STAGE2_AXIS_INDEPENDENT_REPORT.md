# Stage2 independent axis-chain verification

Read-only CPU audit, no model/GPU/Blender render/latent encoding, no changes to existing PERM, data, weights or nju.

## New independent evidence

Installed Blender4.5.1 glTF importer source read (blender_gltf.py:69-94): native glTF Y-up (Dx,Dy,Dz) -> Blender Z-up (Dx,-Dz,Dy). Normals use same rotation; determinant +1. This is not the current SS generator (-Dx,Dz,-Dy), whose determinant is -1. The optional Blender debug_value100 bypass exists but no evidence active; real historical renderer runtime remains missing.

custom_prepare_data.py:74 trimesh.load(force=mesh), :98-103 normalization without axis changes, :164-192 dump same vertices; dual_grid:48-58 same axes into grid. Therefore existing shape supervision is native glTF coordinate space, while renderer normal import uses proper Blender space. These two source paths are not equivalent to one generic dump convention.

CPU two real GLB samples independently checked original trimesh normalized geometry against actual mesh_dumps: 5000 nearest-neighbor sample distances max=median=0, vertex counts equal. NZJ_03_P/0086bbab37559442 native span [.120116,.141267,.999990], dump identical, proper Blender span [.120116,.999990,.141267]. Existing shape support [8,10,64] vs stagedSS [10,64,8]. CR400BF_0503_P/0fab4af3f9a73bdd native [.999990,.170914,.119369], dump identical, Blender [.999990,.119369,.170914], shape [64,12,8] vs SS [12,64,8]. Source SHA in stage2_axis_independent.json. These are geometry comparisons, NOT VAE output or shape20k generation.

Parent whole548 support records read: shape long0=41,long1=0,long2=507; identity IoU median.033996 vs best signed permutations median1/min.932427. This is structural supervision mismatch, not proof the shape20k parameters are damaged or every failure has this sole cause. Parent exact SparseChannel2Spatial parent-child inheritance test constrains all decoded support to descendants of SS cells; withdraw old claim that decoder can globally shift/rotate whole-car support. Decoder can prune/subdivide/local offset, not relocate whole object to disjoint rotated support.

## Up sign: stronger than merely height/depth swap

Actual generator row-vector convention (Rz from source): yaw0 vertices (-Dx,Dz,-Dy) @ Rz(-90) -> q=(-Dz,-Dx,-Dy); cyclic encoder perm yields currentSS=(-Dy,-Dz,-Dx). If native asset Dy is genuine roof/up, SS index0 is DOWN, not UP. At yaw90 currentSS=(-Dy,-Dx,Dz), still down0. Actual native roof/wheel/synthetic marker must independently verify asset semantic up and importer/node transforms; span cannot prove sign.

Proper proposed raw-pipeline frame for native z-long asset: (Dx,-Dz,Dy). This has length rawy and up rawz; formal GLB transform (x,z,-y) returns native (Dx,Dy,Dz), preserving y-up/z-length. As an algebraic conversion from OLD SS it is (-SS2,SS1,-SS0), whose determinant is -1 because old SS already reflected. Native->new raw composition is determinant +1. Do not mislabel oldSS->new as pure rotation. At yaw90 the same algebra yields (-Dz,-Dx,Dy), i.e proper Blender world row Rz(yaw), suitable x-long asset yaw90 canonical variant. This is a hypothesis/design candidate, not a permission to transform cached learned latents directly.

## Minimal correct repair alternatives

A. Keep existing native-frame shape data: derive new SS occupancy from matching shape coords coarsened64 (no old camera/cyclic transform), then reencode SS. This fixes SS/shape support only, NOT canonical output requirement: 507 native z-long become formal GLBy-long,41native x-long remain GLBx-long. Upnativey becomes formalz, likewise wrong expected y-up. It can be a negative-control/compatibility diagnostic, not final accepted pipeline.

B. Preferred shared canonical raw frame: proper nativeglTF->Blender conversion, then per-object proper z-axis rotation to rawlengthy/upz with explicit nose/roof tie definitions. Rebuild/transform geometric meshes, dual grid (coords/local vertex offsets/edge intersection channels), shape latents via encoder, PBR geometry/attributes and SS occupancy from the SAME support+frame. Reencode SS and shape after full geometry alignment; changing tensor spatial axes of an 8-channel SS latent or permuting shape latent coords without features is not mathematically equivalent to reencoding (encoders not assumed equivariant). No unvalidated det-1 mirror as corrective data frame; mirror sign flips affect faces winding/normals/materials.

Existing renders can remain conditions only after proven image/mesh relation, trusted native alpha and camera/view metadata gates. Canonicalization may share target across opposite side input views, but target nose convention needs explicit semantic anchor, not floating-point min-PCA tie alone.

C. Keep fixed shape20k while changing SS: can run tightly scoped gated diagnostic after data contract, but checkpoint trained on mixed native axes, so compatibility with canonical newraw support is not established. No promise that only SS fix suffices or shape reencode is unnecessary. Correct new supervised training corpus must contain matching shape targets; re-training/adapting shape vs using official prior is a later explicit experiment, not Stage2 automatic action. Texture also uses geometry coords, cannot ignore PBR frame alignment.

For smallest sufficient pilot: two asymmetric assets one z-long/one x-long, geometry-only source markers roof/base/nose, source proper determinant and formalexport up/long signs; exact same support SS-vs-shape at64; only after these checks encode a small new-version dataset and decode verify, not full corpus rewrite. Parent prepares marker/image visualization separately; none started here.

## Evidence locations and limits

Remote new alpha_scan/stage2_axis_independent.json contains CPU source-dump distances, hashes, matrices, extents. Installed importer locally downloaded/read blender_gltf_installed.py. Existing parent stage2_support_frames.json copied/read but not rerun. Original actual v4 generation CLI missing; inference from exact parent48map matching and code is strong but not immutable provenance. No renderer invoked; native node-space/world transformation and semantic roof sign require parent asset marker validation. No global PERM/weights edits.

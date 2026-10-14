# Two-asset common-frame pilot

Output: diagnostics/remediation_20261013/common_frame_pilot (new directory). training_approved=false.

CPU only CUDA_VISIBLE_DEVICES='', Open3D voxelization+SS encoder/decoder fp32, torch threads8. No training/shapePBR encoding/inferenceGPU/Blender rendering; old PERM, latent means/features, configs, weights and production data unchanged. Reused new data_toolkit/canonical_frame.py helper after reading it; helper hash recorded. Source native glTF Y-up is convention label, not automatically recognized true roof/nose.

|asset|native->raw yaw|det|new64support|proper transformed oldshape64 vs new mesh IoU|SS encode/decode IoU|
|---|---:|---:|---:|---:|---:|
|0086bbab37559442 NZJ_03_P|0|+1|2033|1.000000|1.000000|
|0fab4af3f9a73bdd CR400BF_0503_P|90|+1|2618|.934614|1.000000|

Each raw mesh length index1=y, up marker native+y -> raw+z; formal GLB lengthz/up+y. Proper geometry vertices/faces stored as NPZ (not textured renderedGLB). Raw spans: first [.120116,.99999,.141267], second [.119369,.99999,.170914]. Formal spans first [.120116,.141267,.99999], second [.119369,.170914,.99999]. Faces unchanged under determinant+1.

New ss_support64 and shape_support64_geometry_only are deliberately the same Open3D mesh voxelization. This is shared construction guarantee, NOT independent shapeVAE latent equality. Existing shape_latents1024 filenames in these examples actually have64 coordinategrid (max<64); only their GEOMETRIC cell centers were transformed using proper helper for support comparison. No latent feature/mean permutation. IoU.9346 may reflect differentvoxelizers/boundary cells, not an automatically bad asset.

Actual SS encoder exists local /data/lzx/TRELLIS/checkpoints/ckpts/ss_enc_conv3d_16l8_fp16,119MB; encoder was NOT found in HF cache. Decoder loaded actual cachedsnapshot /home/test/.cache/huggingface/hub/models--microsoft--TRELLIS-image-large/snapshots/25e0d31ffbebe4b5a97464dd851910efc3002d96/ckpts/ss_dec_conv3d_16l8_fp16. Both architecture configs read; both models convert_to_fp32 + float cpu. Direct commoncoords axis indexing, no oldperm; sample_posterior=False. Each new mean shape8×16³ finite fp32; decodedlogits finite, threshold>0 occupancy IoU1.0. Total model load+twoenc/dec3.934sec, end-to-end command18.352sec. Manifest contains config+actualweight SHA hashes and every source/outputhash.

encoder sha107874eeaa0feb82f51b19db5da7db534fb7e7f19e5a122b9ff1bc2e258bfc6d; decoder sha1c76d4a40519aa2d711cc263a8404105231ac26db31d946bed48b84fee79009a.

COMPLETE.json written after twogeometry and twoCPU VAE cases, includes manifesthash. Process ps query afterward shows no running pilot. Execution was a short remote background shell command rather than managed PID/exit artifact; tool exit0 alone is not a durable process exit record. Final manifest, PILOT_COMPLETE log, actual arrays and absentprocess independently establish completion. No job remains.

Missing acceptance: shape/PBR VAE reencoding and decoding, fixedshape20k compatibility, nativealpha image-camera/nose relation and correct semantic up for assets. SS roundtrip1 proves these two new occupancy inputs survive SS VAE, not trainability/shape mesh quality or whole548data readiness. No permission for training given.

Parent can inspect common_frame_pilot_manifest.json and log locally, remote perasset proper_geometry.npz/ss_support64.npz/shape_support64_geometry_only.npz/new_ss_mean.npz/decoded_ss_support64.npz. Stage4 agents' basic/train ownedfiles untouched.

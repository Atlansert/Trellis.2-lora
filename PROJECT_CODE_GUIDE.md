# TRELLIS.2 项目代码说明文档

> 适用代码根：`/data/lzx/TRELLIS.2`（SSH 别名 `test`）
> 文档性质：**只读代码说明**。本文所有类名、函数名、文件名、参数名、体积、坐标矩阵均来自实地 `ls` / `find` / 读取源码，未读取数据文件。凡未实地确认的内容一律标注"未核实"。
> 相关文档：`REPO_NOTES.md`（代码镜像与未上传目录清单）、`HANDOFF.md`、`HANDOFF_NEXT_20261012.md`、`FULL_AUDIT_20261013.md`、`diagnostics/`（实验证据）。

---

## 1. 项目概览

**一句话定位**：TRELLIS.2 是"单张图片 → 带 PBR 材质的 3D 网格（GLB）"的生成系统；本仓库是上游 `microsoft/TRELLIS.2` 的**微调分支**，在其上做了三阶段 **LoRA 微调**（稀疏结构 SS flow、shape SLat flow 512/1024、texture SLat flow 1024），并配套一整套训练/推理/评测的**契约化安全机制**（检查点完整性、显式权重路径、固定坐标帧验收）。

### 1.1 主要执行链路

```
输入图片 (PNG/JPG/WebP, 可带 alpha)
  │
  ├─ 预处理 preprocess_image: RGBA 直接用 / 否则 BiRefNet 抠图
  │     → 按 alpha 包围盒方形裁剪 (crop_scale) → RGB × alpha (背景=黑) → 最长边 ≤1024
  │
  ├─ 图像条件 (DINOv3 ViT patch tokens + LayerNorm)
  │     cond_512  = get_cond(image, 512)
  │     cond_1024 = get_cond(image, 1024)         # 非 '512' 管线
  │     (可选) SS 专用条件 get_ss_cond: ss_image_cond_model / DINOv2Local@518+crop1.2
  │
  ├─ ① 稀疏结构 SS (Sparse Structure)
  │     噪声 (N, 8, 16, 16, 16)   ← SparseStructureFlowModel(resolution=16, in_channels=8)
  │     → FlowEulerGuidanceIntervalSampler 采样
  │     → sparse_structure_decoder (ss_dec_conv3d_16l8_fp16) > 0
  │     → 64³ 占据 → 按 pipeline_type max_pool 到 32³ (512/1024_cascade/1536_cascade) 或保留 64³ ('1024')
  │     → coords = argwhere(decoded)[:, [0,2,3,4]]  形状 (N, 4) = (batch, x, y, z)
  │     → 可选坐标修正: ss_canonicalize (逐样本最长轴→轴1) 或 ss_swap_yz (coords[:,[0,1,3,2]])
  │
  ├─ ② shape SLat flow
  │     '512'  : sample_shape_slat(cond_512, shape_slat_flow_model_512, coords)      → res 512
  │     '1024' : sample_shape_slat(cond_1024, shape_slat_flow_model_1024, coords)    → res 1024
  │     级联   : sample_shape_slat_cascade(cond_512, cond_1024, 512→1024/1536)
  │                LR 采样 → shape_slat_decoder.upsample(slat, upsample_times=4)
  │                → 量化 coords → token 超 max_num_tokens=49152 则 hr_resolution -= 128 (下限 1024)
  │                → HR 采样 → 反归一化 (std/mean, 32 通道)
  │
  ├─ ③ shape VAE 解码 (dual-grid, FlexiDualGridVaeDecoder)
  │     set_resolution(res) → 输出 7 通道 = 3 顶点偏移 + 3 intersected logits + 1 quad_lerp
  │     → o_voxel.convert.flexible_dual_grid_to_mesh(aabb=[-0.5]³~[0.5]³, grid_size=res)
  │     → Mesh + 子结构 subs
  │
  ├─ ④ texture SLat flow (1024)
  │     shape_slat 重新归一化 → 作为 concat_cond；噪声通道 = tex_in_channels - shape_channels
  │     → tex_slat_decoder(slat, guide_subs=subs) * 0.5 + 0.5
  │
  ├─ ⑤ 组装 MeshWithVoxel
  │     origin=[-0.5,-0.5,-0.5], voxel_size=1/resolution
  │     layout = {base_color: 0:3, metallic: 3:4, roughness: 4:5, alpha: 5:6}
  │     fill_holes()
  │
  └─ ⑥ 导出 o_voxel.postprocess.to_glb
        decimation_target=1000000, texture_size=4096,
        remesh=True, remesh_band=1, remesh_project=0
        → 重网格 / UV / 贴图 → GLB (正式产物)
        调试捷径: --geometry_only → PLY(无 UV) ; --fast_glb → 无纹理 GLB (均不计正式验收)
```

**入口对应关系**：`trellis2/pipelines/trellis2_image_to_3d.py::Trellis2ImageTo3DPipeline.run()` 是①②③④⑤的总编排；`scripts/infer_finetuned.py::infer_single()` 负责⑥与产物落盘。

### 1.2 三条训练线（本项目新增）

| 阶段 | 模型 | 配置示例 | 产物目录示例 |
|---|---|---|---|
| Stage 1 SS | `SparseStructureFlowModel`（官方 1.3B 基座冻结 + LoRA） | `configs/gen/ss_flow_lora_64_views_dinov3_512_50k.json` | `results/multiview_1186_ss_flow_lora_64_views_dinov3_512` |
| Stage 2 shape 512 | `ElasticSLatFlowModel` | `configs/gen/shape_flow_lora_512_train.json` | `results/multiview_1186_flow_512_lora` |
| Stage 2 shape 1024 | `ElasticSLatFlowModel` | `configs/gen/shape_flow_lora_1024_v2_train.json` | `results/multiview_1186_flow_1024_lora_v2` |
| Stage 3 texture 1024 | `ElasticSLatFlowModel` | `configs/gen/tex_flow_lora_1024_train.json` | `results/multiview_1186_tex_flow_lora_1024` |

---

## 2. 目录结构总览

### 2.1 属于"代码"的目录

| 目录 | 职责 | 是否代码 |
|---|---|---|
| `trellis2/` | 核心 Python 包：models / modules / trainers / datasets / pipelines / representations / renderers / utils | ✅ 代码 |
| `scripts/` | 训练与推理启动脚本（`.sh`）、微调推理入口、`utils/` 一次性分析与验证工具 | ✅ 代码 |
| `data_toolkit/` | 数据集准备工具链（下载→dump→体素化→latent 编码→渲染条件→契约构建），含 `blender_script/` | ✅ 代码 |
| `tests/` | 契约与回归测试（10 个 `unittest` 文件） | ✅ 代码 |
| `configs/` | 训练配置 JSON，`gen/` 为生成模型、`scvae/` 为 VAE | ✅ 代码（配置） |
| `train.py` | 训练入口 | ✅ 代码 |
| `run_trellis.2.py` | 批量推理示例（固定 `CUDA_VISIBLE_DEVICES='7'`，读 `../single_side/*.png`） | ✅ 代码 |
| `o-voxel/` | 第三方依赖源码（`o_voxel.postprocess.to_glb`、`o_voxel.convert.flexible_dual_grid_to_mesh`） | ✅ 代码（第三方） |
| `setup.sh` | 环境安装脚本（torch 2.6.0 / flash-attn / CuMesh / o-voxel / FlexGEMM / nvdiffrast） | ✅ 代码（脚本） |
| `README.md` / `REPO_NOTES.md` / `HANDOFF*.md` / `*_AUDIT_*.md` / `REPORT_*.md` / `RECOVERY_STATUS_*.md` / `SECURITY.md` / `LICENSE` | 文档 | ✅ 文档 |
| `diagnostics/` | 实验证据与一次性诊断脚本（`code_repo_export.py`、`verify_comments_only.py` 等）；**体积 57 G**，其中绝大部分为结果文件 | ⚠️ 混合：`.py`/`.md` 属代码与报告，其余为大体积证据 |

### 2.2 数据 / 权重 / 结果目录（**不属于代码**，不随代码镜像上传）

| 目录 | 实测体积 | 说明 / 不上传原因 |
|---|---:|---|
| `nju/` | **561 G** | 另一条训练线（TRELLIS.1）的代码与检查点。子目录：`3d_gen_online`、`3dgen_train`、`hy3d`、`trellis-lora`。**项目规则：只读不改** |
| `results/` | **858 G** | 训练与推理产物（`latest_flow`、`latest_vae`、各 LoRA run 的 `ckpts/`）。⚠️ 是符号链接 → `/data/TRELLIS.2/results` |
| `training_data/` | **179 G** | 训练数据（`multiview_1186/` 下含 `metadata.csv`、`renders_cond`、`renders_cond_t`、`shape_latents`、`pbr_latents`、`ss_latents`、`ss_latents_canon`、`ss_latents_views_canon`、`ss_view_voxels_canon_v4`、`dual_grid_512/1024`、`mesh_dumps`、`pbr_dumps`、`pbr_voxels_1024`、`asset_stats`）。⚠️ 符号链接 → `/data/TRELLIS.2/training_data` |
| `data_for_train/` | **114 G** | 新增训练数据（约 1325 个子目录，每目录为某资产的多视角 PNG） |
| `new_data/` | **126 G** | 待处理/待晋升的新资产包 |
| `new_data_staged/` | 19 M | 暂存区 |
| `new_data_recover_unpromoted/` | 1.6 G | 恢复但未晋升 |
| `new_data_staged_bad/` | 8.6 K | 暂存失败样本 |
| `checkpoints/` | **22 G** | 预训练与微调权重。`ckpts/` 下为 `.json` + `.safetensors`（+ 少量 `.pt`）；`pipeline.json` 为管线装配清单 |
| `results_infer/` | 2.5 G | 推理产物（`ctrl_base_ss_datatest`、`axis10_20k_*`、`data_test_stage1_lora_33k_stage2_lora_20k`、`viewside_*`、`newrun_*`、`B_sweep_001` 等） |
| `rmbg/` | 5.1 G | BiRefNet 抠图权重（第三方） |
| `dinov3/` | 1.2 G | DINOv3 权重（第三方） |
| `trellis2_finetune_release/` | **19 G** | 发布包：含 `trellis2/`、`scripts/`（含 `infer.sh`、`merge_lora_into_base.py`）、`configs/`、`checkpoints/`、`dinov2/`、`dinov3/`、`o-voxel/`、`setup.sh`、`README.md` |
| `assets/` | 19 M | 示例图片、HDRI、App UI 图标 |
| `data_test/` | 217 K | 测试图片（`001.png`…`004.png` 等） |
| `3d_gen_online/` | 105 M | 顶层遗留副本（`nju/3d_gen_online` 同名） |
| `__pycache__/`、`*.pyc` | — | 编译缓存 |

**不上传原因**：以上合计数百 GB，且含第三方权重（`rmbg/`、`dinov3/`）与实验证据；代码镜像只需源码/配置/文档（见 `REPO_NOTES.md`）。因此**克隆代码后不能直接跑通**推理或训练，需自行准备权重与数据。

---

### 2.3 代码镜像（GitHub）实际包含范围

代码镜像按项目所有者确认只保留**最精简主干**（271 个文件 / 约 1.3 MB）：

| 状态 | 内容 |
|---|---|
| ✅ 已上传 | `trellis2/`、`scripts/`、`configs/`、`data_toolkit/`、`tests/`、`train.py`、`run_trellis.2.py`、`LICENSE`、`README.md`、`PROJECT_CODE_GUIDE.md` |
| ❌ 未上传 | `nju/`、`results/`、`results_infer/`、`training_data/`、`data_for_train/`、`new_data*/`、`checkpoints/`、`trellis2_finetune_release/`、`rmbg/`、`dinov3/`、`o-voxel/`、`assets/`、`data_test/`、`3d_gen_online/`、`setup.sh`、`SECURITY.md`、`diagnostics/`、上游 `README.md`，以及交接/审计文档（`HANDOFF*`、`*AUDIT*`、`REPORT_*`、`RECOVERY_STATUS_*`） |

> 注意：本章 2.1 表描述的是**本地项目**的完整结构，其中 `app*.py`、`example*.py`、`o-voxel/`、`setup.sh`、`diagnostics/`、上游 `README.md` **不在**代码镜像中。文中提到这些文件时，仅用于说明本地环境，不代表仓库内容。

---

## 3. 核心数据流与坐标约定

### 3.1 张量 / 分辨率 / 名称对照

| 阶段 | 变量/产物 | 形状 / 分辨率 | 代码位置 |
|---|---|---|---|
| 图像预处理 | `image` (PIL RGBA) | 方形裁剪，最长边 ≤1024，背景=黑 | `trellis2_image_to_3d.py::preprocess_image` |
| 图像条件 | `cond['cond']`, `cond['neg_cond']` | `(B, N_patch, D)`；`neg_cond = zeros_like(cond)` | `::get_cond` / `::get_ss_cond` |
| SS 噪声 | `noise` | `(N, 8, 16, 16, 16)` | `::sample_sparse_structure` |
| SS 潜在 | `z_s` | `(N, 8, 16, 16, 16)`（`flow_model.resolution=16`, `in_channels=8`） | 同上 |
| SS 占据 | `decoded` | 64³ bool；管线可降采样到 32³ | 同上（`decoder(z_s) > 0`，`max_pool3d(ratio)`） |
| SS 稀疏坐标 | `coords` | `(N, 4)` = `(batch, x, y, z)`，int | `torch.argwhere(decoded)[:, [0,2,3,4]]` |
| shape SLat 噪声 | `noise` | SparseTensor，`coords` + `feats (N, in_channels)` | `::sample_shape_slat` |
| shape SLat | `shape_slat` | SparseTensor，32 通道（归一化 mean/std 长度 32） | 同上 + `shape_slat_normalization` |
| 级联 HR 坐标 | `coords` | 由 `shape_slat_decoder.upsample(slat, upsample_times=4)` 量化而来 | `::sample_shape_slat_cascade` |
| shape 网格 | `meshes`, `subs` | `List[Mesh]` + `List[SparseTensor]` | `::decode_shape_slat` |
| 纹理 SLat | `tex_slat` | SparseTensor，32 通道 | `::sample_tex_slat` + `tex_slat_normalization` |
| 纹理体素 | `tex_voxels` | SparseTensor，经 `*0.5+0.5` 到 [0,1] | `::decode_tex_slat` |
| 最终表示 | `MeshWithVoxel` | `voxel_size = 1/resolution`，`origin = [-0.5,-0.5,-0.5]` | `::decode_latent` |
| 导出 | `.glb` / `.mp4` / `.ply` | GLB 为正式产物 | `infer_finetuned.py::infer_single` |

管线档位与 SS 采样分辨率的对应（`run()` 内硬编码字典）：

| `pipeline_type` | SS coords 分辨率 | shape 路径 | 最终 res |
|---|---|---|---|
| `'512'` | 32 | 直接 `shape_slat_flow_model_512` | 512 |
| `'1024'` | 64 | 直接 `shape_slat_flow_model_1024` | 1024 |
| `'1024_cascade'` | 32 | 512 → upsample → 1024 | 1024（token 超限会降级，下限 1024） |
| `'1536_cascade'` | 32 | 512 → upsample → 1536 | ≤1536（按 `max_num_tokens=49152` 逐 128 降级） |

### 3.2 坐标约定（`data_toolkit/canonical_frame.py`，FRAME_VERSION = `trellis.common-frame.gltf-y-up.v1`）

```python
# 1) 原生 glTF Y-up → Blender/pipeline raw Z-up（行列式 +1 的 proper rotation），再绕 yaw
native_gltf_to_internal(v, yaw_degrees):
    base = v[:, [0, 2, 1]] * [1., -1., 1.]     # 即 (x, -z, y)，det = +1
    return base @ Rz(yaw)

# 2) 内部帧 → 正式 GLB 宿主坐标
internal_to_formal_glb(v):
    return v[:, [0, 2, 1]] * [1., 1., -1.]     # 即 (x, z, -y)

# 3) 旧 SS 支撑（索引）→ 内部网格（仅诊断/重编码输入，绝不用于变换 SS 均值）
legacy_ss_support_to_internal(coords, resolution=64):
    out = coords[:, [2, 1, 0]]
    out[:, [0, 2]] = resolution - 1 - out[:, [0, 2]]
    # 语义：index0 = down, index1 = length, index2 = negative depth
```

推理侧同一约定在 `scripts/utils/inference_contract.py` 中实现为：

```python
def glb_coordinates(vertices):
    out = vertices.copy(); out[:,1] = vertices[:,2]; out[:,2] = -vertices[:,1]; return out
# 等价于 (x, y, z) → (x, z, -y)
```

**结论**：
- raw 轴序 = **(depth, length, up)**；raw 的"长度槽"是 **y**（`canonical_frame.py` docstring 原文：*"Raw length slot is y; formal GLB conversion (x,z,-y) makes length z/up y"*）。
- 正式 GLB 宿主坐标 = **(x, z, -y)**：物体长度轴映射到 **z**，上方向映射到 **y**。
- 该文件显式声明：**不会**旋转已生成的网格，**不会**转置已编码的 latent；OBJ/未知 up 轴输入必须另行标定，无隐式回退。

> 未核实项：`trellis2/modules/spatial.py`、`sparse/spatial/*` 内部的体素索引与 (x,y,z) 具体映射；上面所述为数据工具链与推理侧的显式契约层，未展开到每个卷积算子的下标推导。

---

## 4. 模块逐个说明

### 4.1 `trellis2/models/` — 网络结构

**职责**：定义所有可训练/可加载的模型类，并提供 `from_pretrained`（严格键校验，失败不回退）。

| 文件 | 关键类 / 函数 | 作用 |
|---|---|---|
| `__init__.py` | `__getattr__`（延迟导入）、`from_pretrained(path)`、`CheckpointValidationError` | 模型工厂。导出 `SparseStructureEncoder/Decoder`、`SparseStructureFlowModel`、`SLatFlowModel`、`ElasticSLatFlowModel`、`SparseUnetVaeEncoder/Decoder`、`FlexiDualGridVaeEncoder/Decoder`。`from_pretrained` 要求 `{path}.json` + `{path}.safetensors` 成对存在，且**剔除 `rope_phases` 后**严格比对键；记录 `model._checkpoint_source`（`config_file`/`model_file`/`requested`/`computed_buffers_regenerated`）供推理 manifest 溯源 |
| `sparse_structure_flow.py` | `SparseStructureFlowModel`、`TimestepEmbedder` | SS 的 DiT。`resolution=16`、`in_channels=8`、`cond_channels=1024`、`num_blocks=30`、`num_heads=12`、`model_channels=1536`、`pe_mode='rope'`、`share_mod=True`、`initialization='scaled'`、`qk_rms_norm=True`、`qk_rms_norm_cross=True`。`forward(x,t,cond)` 断言输入形状必须等于 `(B, in_channels, R, R, R)`；`rope_phases` 为注册 buffer（16³ × 64 = 262144 元素） |
| `sparse_structure_vae.py` | `SparseStructureEncoder`、`SparseStructureDecoder`、`ResBlock3d`、`DownsampleBlock3d`、`UpsampleBlock3d`、`norm_layer` | SS 的 3D VAE。编码器输出 `mean/logvar`（`latent_channels*2`）；解码器重建占据 |
| `structured_latent_flow.py` | `SLatFlowModel`、`ElasticSLatFlowModel` | 稀疏 latent 的 DiT。`forward(x, t, cond, concat_cond=None)`：有 `concat_cond` 时 `sp.sparse_cat([x, concat_cond], dim=-1)`（纹理 flow 用 shape SLat 作拼接条件）；`ElasticSLatFlowModel = SparseTransformerElasticMixin + SLatFlowModel`（低显存弹性） |
| `sparse_elastic_mixin.py` | `SparseTransformerElasticMixin` | 弹性显存调度混入 |
| `sc_vaes/sparse_unet_vae.py` | `SparseUnetVaeEncoder`、`SparseUnetVaeDecoder`、`SparseResBlock3d`、`SparseConvNeXtBlock3d`、`SparseResBlockDownsample3d/Upsample3d/S2C3d/C2S3d` | 稀疏 UNet VAE 主干（SC-VAE） |
| `sc_vaes/fdg_vae.py` | `FlexiDualGridVaeEncoder`、`FlexiDualGridVaeDecoder` | 双网格（flexible dual grid）VAE。解码器 7 输出通道：`[0:3]` 顶点偏移（`(1+2*margin)*sigmoid(f) - margin`）、`[3:6]` intersected logits、`[6:7]` `softplus` quad_lerp；经 `o_voxel.convert.flexible_dual_grid_to_mesh` 出网格；`set_resolution(res)` 决定 512/1024 |

### 4.2 `trellis2/modules/` — 基础算子与 LoRA

**职责**：稀疏/稠密注意力、位移编码、卷积后端、LoRA 注入、图像特征抽取。

| 文件 | 关键类 / 函数 | 作用 |
|---|---|---|
| `lora.py` | `LoRALinear`、`apply_lora_to_linear_layers(model, target_keywords, rank=8, alpha=8.0, dropout=0.0)`、`_parent` | 见 §5.2 |
| `image_feature_extractor.py` | `DinoV2FeatureExtractor`、`DinoV3FeatureExtractor` | 条件编码器。DINOv3：`transformers.DINOv3ViTModel.from_pretrained(model_name)`，`transform` 用 ImageNet mean/std 归一化，返回 `x_prenorm` 后 `F.layer_norm` 的 patch tokens；`image_size` 可被 `get_cond` 覆写 |
| `attention/modules.py` | `MultiHeadAttention`（稠密版） | 稠密注意力 |
| `attention/full_attn.py` | `scaled_dot_product_attention` | 稠密全局注意力 |
| `attention/rope.py` | `RotaryPositionEmbedder` | 稠密 RoPE，产生 `rope_phases` buffer |
| `attention/config.py` | 配置常量 | 注意力后端配置 |
| `sparse/basic.py` | `SparseTensor`（+ `sparse_cat` 等） | **稀疏张量的核心数据结构**（`feats` + `coords`，`replace()` 派生） |
| `sparse/linear.py` | `SparseLinear` | 稀疏线性层（`SLatFlowModel` 的输入/输出层） |
| `sparse/norm.py` | 稀疏归一化 | — |
| `sparse/nonlinearity.py` | 稀疏激活 | — |
| `sparse/spatial/basic.py` | `SparseDownsample`、`SparseUpsample` | 稀疏下/上采样（`subdivision` 参数） |
| `sparse/spatial/spatial2channel.py` | `SparseSpatial2Channel`、`SparseChannel2Spatial` | 空间↔通道重排 |
| `sparse/attention/modules.py` | `SparseMultiHeadAttention`、`SparseMultiHeadRMSNorm` | **LoRA 目标的宿主**：`to_qkv = nn.Linear(channels, channels*3)`、`to_q = nn.Linear(channels, channels)`、`to_kv = nn.Linear(ctx_channels, channels*2)`、`to_out = nn.Linear(channels, channels)`、`q_rms_norm`/`k_rms_norm`；`forward(x, context=None)` 走 `_fused_pre` / `_linear` / `_reshape_chs` |
| `sparse/attention/full_attn.py` | `sparse_scaled_dot_product_attention`（多分派重载） | 稀疏 SDPA：支持 qkv 合一、q+kv、q/k/v 分离、`VarLenTensor` 与稠密 mixed |
| `sparse/attention/windowed_attn.py` | `calc_window_partition`、`sparse_windowed_scaled_dot_product_self_attention`、`..._cross_attention` | 窗口化稀疏注意力 |
| `sparse/attention/rope.py` | `SparseRotaryPositionEmbedder` | 稀疏 RoPE |
| `sparse/conv/*.py` | `SparseConv3d`、`SparseInverseConv3d`（`conv.py`）；后端实现：`conv_spconv.py`、`conv_flex_gemm.py`、`conv_torchsparse.py`、`config.py` | 稀疏卷积三后端 |
| `sparse/transformer/blocks.py` | `SparseFeedForwardNet`、`SparseTransformerBlock`、`SparseTransformerCrossBlock` | 稀疏 Transformer 块 |
| `sparse/transformer/modulated.py` | `ModulatedSparseTransformerBlock`、`ModulatedSparseTransformerCrossBlock` | **SLat flow 的实际 block**：持 `self.self_attn`、`self.cross_attn`、`self.mlp`、`self.adaLN_modulation`（`chunk(6)` 得 shift/scale/gate） |
| `transformer/blocks.py` | `AbsolutePositionEmbedder`、`FeedForwardNet`、`TransformerBlock`、`TransformerCrossBlock` | 稠密 Transformer 块 |
| `transformer/modulated.py` | `ModulatedTransformerBlock`、`ModulatedTransformerCrossBlock` | **SS flow 的实际 block**（同样持有 `self_attn`/`cross_attn`/`mlp`/`adaLN_modulation`） |
| `norm.py` | `GroupNorm32`、`ChannelLayerNorm32` | 归一化层 |
| `spatial.py` | `pixel_shuffle_3d` | 3D 像素重排（VAE 上采样用） |
| `utils.py` | `zero_module`、`convert_module_to`、`convert_module_to_f16/f32`、`manual_cast`、`str_to_dtype` | 类型/初始化工具 |

### 4.3 `trellis2/trainers/` — 训练器

**职责**：通用训练循环 + 曲线/检查点/EMA/混合精度 + 任务特化。

| 文件 | 关键类 / 函数 | 作用 |
|---|---|---|
| `basic.py` | `BasicTrainer` | 见 §5.1（`__init__` / `load` / `save` / `finetune_from` / `run` / `run_step` / `snapshot` / `update_ema` / `check_ddp` / `save_logs` / `check_abort` / `profile`） |
| `utils.py` | `str_to_dtype`、`make_master_params`、`unflatten_master_params`、`model_params_to_master_params`、`master_params_to_model_params`、`model_grads_to_master_grads`、`zero_grad`、`LinearWarmupLRScheduler` | 主参数（master params）机制与 LR 调度 |
| `flow_matching/flow_matching.py` | `FlowMatchingTrainer`、`FlowMatchingCFGTrainer`、`TextConditionedFlowMatchingCFGTrainer`、`ImageConditionedFlowMatchingCFGTrainer` | flow matching 训练：`diffuse` / `reverse_diffuse` / `get_v` / `sample_t` / `training_losses` / `get_sampler` / `run_snapshot` |
| `flow_matching/sparse_flow_matching.py` | `SparseFlowMatchingTrainer`、`SparseFlowMatchingCFGTrainer`、`TextConditioned...`、`ImageConditioned...`、`MultiImageConditionedSparseFlowMatchingCFGTrainer` | 稀疏 latent 版本（§7 提到的 `num_samples = min(...)` 补丁在此文件的 `run_snapshot`） |
| `flow_matching/mixins/image_conditioned.py` | `DinoV2FeatureExtractor`、`DinoV3FeatureExtractor`、`DinoV2LocalFeatureExtractor`、`ImageConditionedMixin`、`MultiImageConditionedMixin` | 条件编码器（训练侧）+ `get_cond`/`get_inference_cond`/`vis_cond` |
| `flow_matching/mixins/classifier_free_guidance.py` | `ClassifierFreeGuidanceMixin` | `p_uncond` 训练期 dropout、`get_inference_cond` 生成 `neg_cond`、`get_sampler` 返回 `FlowEulerGuidanceIntervalSampler` |
| `flow_matching/mixins/text_conditioned.py` | `TextConditionedMixin` | CLIP 文本条件（本项目未用于训练） |
| `vae/shape_vae.py` | `ShapeVaeTrainer` | shape VAE 微调（渲染监督：`_randomize_camera`、`_render_batch`、`loss_utils`） |
| `vae/pbr_vae.py` | `PbrVaeTrainer` | PBR VAE 微调 |
| `vae/sparse_structure_vae.py` | `SparseStructureVaeTrainer` | SS VAE 微调 |
| `__init__.py` | `__getattr__` | 训练器工厂（`getattr(trainers, cfg.trainer.name)`） |

### 4.4 `trellis2/datasets/` — 数据集

**职责**：从磁盘读取 latent/渲染条件，构造训练样本；含本项目的规范性（canonical）契约。

| 文件 | 关键类 / 函数 | 作用 |
|---|---|---|
| `components.py` | `StandardDatasetBase`、`ImageConditionedMixin`、`MultiImageConditionedMixin` | 基类：`filter_metadata` / `get_instance` / `__getitem__` / `collate_fn`；条件混入（`image_size`、`real_view_prob`、`max_image_cond_view`） |
| `sparse_structure_latent.py` | `SparseStructureLatent`、`ImageConditionedSparseStructureLatent`、`SparseStructureLatentVisMixin` | SS latent 数据集（`decode_latent`、`visualize_sample`） |
| `sparse_structure_latent_views.py` | `ImageConditionedSparseStructureLatentViews` | **SS LoRA 的实际数据集**。TRELLIS.1 式成对监督：一张渲染视角 ↔ 该视角相机帧下的 SS latent。关键参数：`image_size`、`min_aesthetic_score`、`crop_scale`（必须 ≥1.0）、`num_views=24`、`latent_frame ∈ {'view','canonical'}`、`canonical_latent_root`、`min_view_aspect`、`max_view_offaxis`（需要 `renders_cond/_view_offaxis.json`）、`canonical_manifest` + `allow_staged_contract`、`skip_snapshot=True`（默认关闭快照） |
| `structured_latent.py` | `SLat`、`ImageConditionedSLat`、`SLatVisMixin` | 通用 SLat 数据集 |
| `structured_latent_shape.py` | `SLatShape`、`ImageConditionedSLatShape`、`MultiImageConditionedSLatShape` | **shape flow 数据集**（含 `normalization`、`max_tokens`、`real_view_prob`） |
| `structured_latent_svpbr.py` | `SLatPbr`、`ImageConditionedSLatPbr` | **texture flow 数据集**（`full_pbr`、`attrs`、`shape_slat_normalization`、`pbr_slat_normalization`） |
| `sparse_voxel_pbr.py` | `SparseVoxelPbrDataset` | PBR VAE 数据集（读 `mesh`+PBR 体素） |
| `flexi_dual_grid.py` | `FlexiDualGridDataset` | 双网格 shape VAE 数据集 |
| `canonical_contract.py` | `load_canonical_contract(manifest_path, canonical_root, allow_staged=False)`、`file_sha256` | **canonical 数据契约校验**（stdlib only）：校验 `COMPLETE.json` → manifest sha256 → schema `trellis.canonical-contract.v1` → canonical root 归属 → `TRAINING_APPROVAL.json`（含 `coordinate_evidence`/`mask_evidence` 的 path+sha256+`passed`）→ 每个 object 的 `transforms.json`/`metadata.csv`/`renders_cond_t/_view_offaxis.json`/`canonical/{name}.npz`/每视角 image+voxel+latent 的 sha256 与帧序号。staged 数据默认拒绝 |

### 4.5 `trellis2/pipelines/` — 推理管线

| 文件 | 关键类 / 函数 | 作用 |
|---|---|---|
| `base.py` | `Pipeline`（`from_pretrained(path, config_file="pipeline.json")`、`to/cuda/cpu`、`device`） | 按 `pipeline.json` 的 `args.models` 逐组件加载；本地点缺失时**直接 `FileNotFoundError`，绝不回退**；支持显式跨仓库组件（`len(v.split('/')) >= 3`）；`Trellis2ImageTo3DPipeline.model_names_to_load` 决定加载哪些组件 |
| `trellis2_image_to_3d.py` | `Trellis2ImageTo3DPipeline` | §1.1 全链路。方法：`preprocess_image`、`get_cond`、`get_ss_cond`、`sample_sparse_structure`、`sample_shape_slat`、`sample_shape_slat_cascade`、`decode_shape_slat`、`sample_tex_slat`、`decode_tex_slat`、`decode_latent`、`run`。`from_pretrained` 额外读取 `ss_image_cond_model` / `ss_swap_yz` / `ss_canonicalize` |
| `trellis2_texturing.py` | `Trellis2TexturingPipeline` | 只做贴图：`preprocess_mesh`、`encode_shape_slat`、`sample_tex_slat`、`decode_tex_slat`、`postprocess_mesh`、`run` |
| `samplers/base.py` | `Sampler`（抽象 `sample`） | 采样器基类 |
| `samplers/flow_euler.py` | `FlowEulerSampler`、`FlowEulerCfgSampler`、`FlowEulerGuidanceIntervalSampler` | 欧拉 flow 采样；`_pred_to_xstart` / `_v_to_xstart_eps` / `sample_once` / `sample` |
| `samplers/classifier_free_guidance_mixin.py` | `ClassifierFreeGuidanceSamplerMixin` | CFG：`_inference_model(..., guidance_strength, guidance_rescale)` |
| `samplers/guidance_interval_mixin.py` | `GuidanceIntervalSamplerMixin` | 只在 `guidance_interval` 区间施加 CFG |
| `rembg/BiRefNet.py` | `BiRefNet`（`model_name="ZhengPeng7/BiRefNet"`） | 抠图模型，`__call__(image) -> PIL` |

### 4.6 `trellis2/representations/` — 结果表示

| 文件 | 关键类 | 作用 |
|---|---|---|
| `mesh/base.py` | `Mesh`、`Texture`、`PbrMaterial`、`MeshWithPbrMaterial`、`MeshWithVoxel`、`TextureFilterMode`、`TextureWrapMode`、`AlphaMode` | `Mesh`：`fill_holes(max_hole_perimeter=3e-2)`、`remove_faces`、`simplify(target, options)`（推理里 `mesh.simplify(16777216)` 为 nvdiffrast 上限）。`MeshWithVoxel`：带 `attrs`/`coords`/`layout`/`voxel_size`，`query_attrs` / `query_vertex_attrs` |
| `voxel/voxel_model.py` | `Voxel` | `position`、`split_attrs`、`save`、`load` |

### 4.7 `trellis2/renderers/` — 渲染器（可视化）

| 文件 | 关键类 / 函数 | 作用 |
|---|---|---|
| `mesh_renderer.py` | `MeshRenderer`、`intrinsics_to_projection` | 几何/法线渲染 |
| `pbr_mesh_renderer.py` | `PbrMeshRenderer`、`EnvMap`、`latlong_to_cubemap`、`cube_to_dir`、`screen_space_ambient_occlusion`、`aces_tonemapping`、`gamma_correction`、`intrinsics_to_projection` | PBR 着色 + 环境贴图（`render_utils.make_pbr_vis_frames` 用它出视频帧） |
| `voxel_renderer.py` | `VoxelRenderer` | 稀疏体素渲染 |
| `__init__.py` | `__getattr__` | 延迟导出 |

### 4.8 `trellis2/utils/` — 工具（本项目安全契约的落点）

| 文件 | 关键函数 / 类 | 作用 |
|---|---|---|
| `checkpoint_contract.py` | `marker_name`、`atomic_save_group`（别名 `save_checkpoint_group`）、`validate_checkpoint`（别名 `validate_complete_checkpoint`）、`list_complete_steps`、`cpu_snapshot`、`cast_model_preserving_lora_and_complex`、`align_optimizer_moments`、`capture_rng_state`、`restore_rng_state`、`snapshot_allowed`、`SCHEMA='trellis.checkpoint-group.v1'` | 见 §5.3 |
| `train_launch_contract.py` | `expected_files`、`resolve_checkpoint`、`preserve_launch_records`、`run_attempts`、`load_pretrained_checked`、`_has_history` | 见 §5.4 |
| `data_utils.py` | `recursive_to_device`、`load_balanced_group_indices`、`cycle`、`ResumableSampler`、`BalancedResumableSampler` | 数据加载与可恢复采样器（`state_dict`/`load_state_dict`） |
| `dist_utils.py` | `setup_dist`、`read_file_dist`、`unwrap_dist`、`master_first`、`local_master_first` | 分布式工具 |
| `elastic_utils.py` | `MemoryController`、`LinearMemoryController`、`ElasticModule`、`ElasticModuleMixin`（`with_mem_ratio`） | 弹性显存 |
| `general_utils.py` | `dict_merge`、`dict_foreach`、`dict_reduce`、`dict_flatten`、`nested_contexts`、`make_grid`、`text_image`、`save_image_with_notes`、`atol`、`rtol`、`indent` | 通用工具；可视化拼图 |
| `grad_clip_utils.py` | `AdaptiveGradClipper` | 自适应梯度裁剪（config 里 `max_norm=1.0, clip_percentile=95`） |
| `loss_utils.py` | `smooth_l1_loss`、`l1_loss`、`l2_loss`、`ssim`、`psnr`、`lpips`、`normal_angle` | VAE 训练损失 |
| `mesh_utils.py` | `read_ply`、`write_ply`、`write_pbr_ply` | 网格 IO |
| `random_utils.py` | `radical_inverse`、`halton_sequence`、`hammersley_sequence`、`sphere_hammersley_sequence` | 视角采样 |
| `render_utils.py` | `yaw_pitch_r_fov_to_extrinsics_intrinsics`、`get_renderer`、`render_frames`、`render_video`、`render_multiview`、`render_snapshot`、`make_pbr_vis_frames` | 出视频/快照 |
| `vis_utils.py` | `pca_color`、`vis_sparse_tensor` | latent 可视化 |

### 4.9 `scripts/` — 启动与分析脚本

**职责**：训练启动 `.sh`、微调推理入口 `infer_finetuned.py`、PBR 管线准备 `prepare_pbr_pipeline.py`，以及 `utils/` 下大量一次性分析与验证工具。

| 文件 / 目录 | 作用 |
|---|---|
| `infer_finetuned.py` | **微调推理入口**，见 §6.1 |
| `prepare_pbr_pipeline.py` | PBR 数据管线：`load_meta`、`finished`、`stage_a`（dump）、`vox_one`、`align_to_dual_grid`、`fix_dump`、`stage_b`（体素化）、`stage_c(gpu='5')`（编码）、`stage_d`
| `run_ss_lora_*.sh`（8 个） | SS LoRA 训练启动脚本，命名即实验口径，例如 `run_ss_lora_dinov3_512_50k.sh`、`run_ss_lora_canon_axis10_dinov3_20k.sh` |
| `run_stage2_*.sh`、`run_tex_flow_lora.sh` | shape/texture LoRA 训练启动 |
| `run_ss_lora_64_resume_1000k.sh`、`run_stage2_lora_v2_resume_200k.sh` | 续训（`--load_dir` + `--ckpt latest`） |
| `infer_ss33k_shape20k_texN_gpu.sh`、`infer_ss_dinov3_sweep_gpu.sh`、`infer_ss_sweep_generic.sh`、`infer_stage1_028000_stage2_N_gpu.sh` | 推理批处理/扫描，含"已完成则跳过"与 GLB 计数校验 |
| `training_history/`、`inference_history/` | 历史脚本归档（已不用于新实验） |
| `results_infer/` | 脚本运行日志（`_sweep_watch.log`） |
| `utils/` | 见下 |

`scripts/utils/` 分组（按用途）：

| 分组 | 文件 | 作用 |
|---|---|---|
| 契约与验收 | `inference_contract.py`、`evaluate_fixed_frame.py`、`mesh_metrics.py`、`validate_glb.py`、`final_verify.py`、`final_table.py` | §6 |
| 环境/代码补丁 | `apply_patches.py`、`convert_safetensors_to_pt.py` | 见 §11 |
| 坐标/帧诊断 | `canon_transform_search.py`、`canon_axis_calib{,2,3}.py`、`canon_axis_agreement.py`、`canon_aspect_bins.py`、`canon_chirality.py`、`canon_pose_offset.py`、`canon_sign_calib.py`、`ss_frame_probe.py`、`ss_yz_swap_test.py`、`ss_orient_datatest.py`、`ss_orient_oldrun.py`、`ss_eval_canon.py`、`verify_canon_data.py`、`verify_canon_full.py` | canonical / SS 朝向标定与验证 |
| SS 行为探针 | `ss_cond_ab.py`、`ss_cond_dtype_ab.py`、`ss_crop_probe.py`、`ss_occ_diag.py`、`ss_prior_probe.py`、`ss_seed_probe.py` | SS 条件、dtype、裁剪、占据、种子敏感性 |
| 数据/渲染 | `render_views.py`、`render_new_data.py`、`build_renders_cond_t.py`、`verify_renders_cond_t.py`、`derive_alpha.py`、`stage_new_data.py`、`promote_new_data.py`、`ascii_silhouette.py` | 渲染与数据晋升 |
| 展示 | `make_preview.py`、`make_compare.py`、`sweep_report.py` | 拼图与扫描报表 |
| 监视 | `sweep_watch.sh`、`sweep_watch_canonical.sh`、`probe_watch.sh` | 轮询脚本 |

> `scripts/utils/` 下共 50 余个文件，上表按用途归类；**未逐个细读其内部实现**（多为一次性实验工具），仅确认存在与 CLI 参数。

### 4.10 `data_toolkit/` — 数据工具链

见 §7。

### 4.11 `tests/` — 测试

见 §8。

### 4.12 `configs/` — 训练配置

**职责**：`train.py` 的 `--config` 输入；JSON 结构固定为 `models` / `pretrained` / `lora` / `dataset` / `trainer` 五个顶层键（`pretrained` 与 `lora` 可选）。

| 子目录 | 文件数 | 内容 |
|---|---|---|
| `configs/gen/` | 30 | 生成模型（SS flow / shape SLat flow / texture SLat flow），含官方全量微调配置与 LoRA 配置 |
| `configs/scvae/` | 6 | VAE 微调：`shape_vae_next_dc_f16c32_fp16{,_ft_512}.json`、`tex_vae_next_dc_f16c32_fp16{,_ft_512}.json`、`shape_vae_ft_train.json` |

---

## 5. 训练链路

### 5.1 入口 `train.py`

```text
python train.py --config <cfg.json> --output_dir <dir> [--load_dir <dir>] [--ckpt latest|none|<step>]
                --data_dir '<json 根描述>' --num_gpus N [--auto_retry 3] [--tryrun] [--profile]
                [--num_nodes/--node_rank/--master_addr/--master_port]
```

流程（`__main__` 与 `main()`）：

1. 读 config，与命令行合并为 `cfg`（`easydict`）。
2. **写盘前**先 `find_ckpt(cfg)` → `resolve_checkpoint(cfg)`（安全性校验，见 §5.4）。
3. `num_gpus < 1` 直接 `ValueError`（"Training requires at least one explicitly available GPU"）。
4. rank0 调 `preserve_launch_records(cfg, config, command, __file__)` 写 `config.json` / `command.txt` / `run_attempts/{launch_id}.json`，得到 `cfg.launch_id`。
5. `run_attempts(cfg, runner, ...)` 包装 `mp.spawn(main, nprocs=num_gpus)`；**单卡走 `main(0, cfg)`**。
6. `main()`：`setup_dist`（world_size>1）→ `setup_rng(rank)`（`torch.manual_seed(rank)` 等）→ 构建 dataset（`getattr(datasets, cfg.dataset.name)(cfg.data_dir, **cfg.dataset.args)`）→ 构建模型（`getattr(models, model.name)(**model.args).cuda()`）。
7. `cfg.pretrained` 存在时逐项加载（`.safetensors` 用 `safetensors.torch.load_file`，否则 `torch.load(..., weights_only=True)`），调 `load_pretrained_checked`（剔除 `rope_phases`，其余键严格比对，不符即 `RuntimeError`）。
8. `cfg.lora.enabled` 为真：先 `param.requires_grad = False` 冻结全部参数，再 `apply_lora_to_linear_layers(...)`；`replaced` 为空则 `RuntimeError('LoRA enabled for X, but no target layers matched')`。
9. rank0 打印每个 backbone 的参数量，并把完整参数表写入 `output_dir/run_attempts/{launch_id}_attempt{attempt_index}_{name}_model_summary.txt`（以 `'x'` 模式创建，已存在会抛错）。
10. 构建 trainer：`getattr(trainers, cfg.trainer.name)(model_dict, dataset, **cfg.trainer.args, output_dir=..., load_dir=..., step=cfg.load_ckpt)`。
11. `--tryrun` 则只构建；否则 `trainer.run()`（或 `--profile` 则 `trainer.profile()`）。

### 5.2 LoRA 注入 `trellis2/modules/lora.py`

```python
class LoRALinear(nn.Module):
    # weight / bias：detach().clone() 且 requires_grad=False（冻结基线）
    # lora_down: Linear(in_features → rank, bias=False, dtype=torch.float32)
    # lora_up  : Linear(rank → out_features, bias=False, dtype=torch.float32)
    # init: kaiming_uniform_(lora_down, a=sqrt(5)) / zeros_(lora_up)
    # scale = alpha / rank
    def forward(self, x):
        base   = F.linear(x.to(dtype=self.weight.dtype), self.weight, self.bias)
        update = self.lora_up(self.dropout(self.lora_down(x.float())))
        return base + update.to(dtype=base.dtype) * self.scale
```

要点：
- adapter 恒为 **fp32**（即使基座是 bf16），forward 时把 base 输出对齐回 `base.dtype`，因此 AMP/bf16 语义不变。
- `apply_lora_to_linear_layers(model, target_keywords, rank=8, alpha=8.0, dropout=0.0)`：
  - 匹配方式是 `isinstance(module, nn.Linear) and any(k in name for k in targets)`（**子串匹配**）。
  - **幂等保护**：先收集已存在 `LoRALinear` 的限定名，`if any(prefix == '' or name == prefix or name.startswith(prefix + '.'))` 则跳过整棵子树 —— 防止二次调用在 adapter 内部再套 adapter（会污染检查点结构）。

**实测 LoRA 规模（从 `results/multiview_1186_ss_flow_lora_64_views/ckpts/denoiser_step0033000.pt` 读取统计）**：

| 项 | 数值 |
|---|---|
| 目标层数 | **150** = 30 blocks × 5 targets |
| adapter 张量数 | **300**（每层 `lora_down.weight` + `lora_up.weight`） |
| 可训练参数量 | **4,669,440** |
| 基座参数量 | **1,296,849,416** |
| 检查点内全部张量元素 | 1,297,111,560（差值 262,144 = `rope_phases` buffer：16³ × 64） |

逐目标参数（rank=8，`channels=1536`，`ctx_channels=1024`）：

| 目标 | down | up | 每块合计 | ×30 |
|---|---:|---:|---:|---:|
| `self_attn.to_qkv` | 8×1536 = 12,288 | 8×4608 = 36,864 | 49,152 | 1,474,560 |
| `self_attn.to_out` | 12,288 | 12,288 | 24,576 | 737,280 |
| `cross_attn.to_q` | 12,288 | 12,288 | 24,576 | 737,280 |
| `cross_attn.to_kv` | 8×1024 = 8,192 | 8×3072 = 24,576 | 32,768 | 983,040 |
| `cross_attn.to_out` | 12,288 | 12,288 | 24,576 | 737,280 |
| **合计** | | | **155,072** | **4,669,440** |

**推理侧的严格校验** `scripts/utils/inference_contract.py::validate_lora(model, replaced, state, rank, alpha, dropout)`：
1. `rank > 0`、`alpha` 有限且 >0、`0 <= dropout < 1`；
2. `len(replaced) == 150` 且去重后仍为 150；
3. 期望 adapter key 集合 `{name}.lora_down.weight` / `{name}.lora_up.weight` 必须与 state 中实际 `.lora_down.`/`.lora_up.` 键**完全相等**；
4. 每个 replaced 名称必须包含合法的 5 个 target 之一；
5. 运行时 `layer.rank`/`layer.alpha` 必须等于传入值；
6. `down.shape == (rank, in_features)`、`up.shape == (out_features, rank)`。

### 5.3 检查点保存 / 恢复契约（`trellis2/trainers/basic.py` + `utils/checkpoint_contract.py`）

**组文件命名**（`ckpts/` 下）：

```
{model_name}_step{step:07d}.pt            # 每个 model_dict key 一个
{model_name}_ema{rate}_step{step:07d}.pt  # 每个 ema_rate 一个
misc_step{step:07d}.pt                    # 优化器/调度器/RNG/采样器状态
complete_step{step:07d}.json              # 唯一的"提交成功"标记
```

`misc` 内容：`optimizer`、`step`、`data_sampler`、`rng_state`、以及按需的 `scaler`（amp+fp16）/ `log_scale`（inflat_all+fp16）/ `lr_scheduler` / `elastic_controller` / `grad_clip`。

**提交语义** `atomic_save_group(ckpt_dir, step, payloads, metadata=None, non_blocking=True)`：
- `non_blocking` 仅为兼容参数，**永远同步完成**（"no writer threads are used"）。
- 先 `_check_names`：文件名必须唯一、`Path(name).name == name`、无 `/` `\`、必须以 `_step{step:07d}.pt` 结尾；**必须包含 `misc_step...pt`，且必须有非 `misc_` 的权重文件**。
- 目标文件或 marker 已存在 → `FileExistsError`（"preserve old/partial files and use a new run directory"）—— **不覆盖、不清理中途失败的文件**。
- `.checkpoint_step{step:07d}.reservation` 独占锁（`open(..., 'x')` + fsync），防止并发写同一 step。
- 每个载荷先 `cpu_snapshot()`（自有 CPU 拷贝，**不与 live optimizer/EMA 共享存储**），写 `.name.token.tmp` → `flush` + `fsync` + `os.replace`，再记录 `{name, size_bytes, sha256}`。
- 最后一个原子写入 `complete_step*.json`（schema `trellis.checkpoint-group.v1`，含 `files`、`metadata`、`synchronous: True`、`requested_non_blocking`），并 `_fsync_dir`。
- **可见性判定点只有一个：complete marker**。`misc` 单独存在不构成提交。

**校验**：`validate_checkpoint(ckpt_dir, step, verify_hashes=True)` 检查 schema/step、文件名规则、每文件 `size_bytes > 0`、非符号链接、size 一致、sha256（可选）一致；`list_complete_steps(ckpt_dir, verify_hashes=False)` 遍历 `complete_step*.json` 并逐个校验，返回排序去重后的 step 列表。

**加载** `BasicTrainer.load(load_dir, step)`：
- 期望名集合 = 所有 model 的 `_step` + 所有 `ema_rate` 的 `_ema{rate}_step` + `misc`；有 marker 则必须与 marker 内文件集合**完全一致**（"checkpoint complete group does not match configured models/EMA rates"）。
- 无 marker 时：**只有调用方显式选择了数字 step** 才允许 legacy 加载，且要求全部文件非空，并 `RuntimeWarning`；否则 `ValueError('legacy checkpoint group incomplete')`。
- 每个模型加载前 `model_ckpt.pop('rope_phases')`，保留新初始化的 complex64 RoPE；缺失键除 `rope_phases` 外一律 `RuntimeError`。
- `misc['step'] != step` → `ValueError`；RNG 恢复只对 rank0 生效并 warn。
- amp 模式加载后调 `cast_model_preserving_lora_and_complex(model, dtype)`：把非 LoRA 基座转 dtype，但**保持 LoRA adapter 精确 fp32 与 Parameter 身份**（若 dtype 转换替换了 Parameter 身份则 `RuntimeError`），并保留 complex buffer。

**保存** `BasicTrainer.save()`：`assert self.is_master`；由 master params 与 ema params 生成各自 state_dict，组 payloads 后调 `atomic_save_group`；成功后记录 `self._last_committed_step = self.step`。

**`run()` 收尾契约**：当 `max_steps` 不与 `i_save` 对齐时，循环结束后若 `_last_committed_step != self.step`，则（已存在 marker 就校验，否则）**强制补一次 `save()`**，保证最终 step 一定有已提交检查点；随后 `snapshot(suffix='final')`（受 `snapshot_allowed` 控制）。

### 5.4 启动安全契约（`utils/train_launch_contract.py`）

- `resolve_checkpoint(cfg)`（**在任何写盘之前**调用）：
  - `--load_dir` 与 `--output_dir` 不同且 output 已有历史（`config.json`/`command.txt`/`ckpts/` 非空）→ `ValueError`（禁止把外部检查点混进已有输出目录）。
  - `--ckpt none`：output 有历史则拒绝。
  - `--ckpt latest`：取 `list_complete_steps` 的最大值并 `validate` + `_require_files`；无 complete 但有历史 → `ValueError`（明确写："legacy misc is NOT a commit"）；load 目录全新空目录 → 打印 "latest resolves to step0"。
  - `--ckpt <正整数>`：有 marker 则严格校验；无 marker 则 `RuntimeWarning`（显式 legacy），仅检查文件名/大小完整性。
- `expected_files(cfg, step)`：由 `cfg['models']` 与 `trainer.args.ema_rate` 推导期望文件清单（默认 `ema_rate=0.9999`）。
- `preserve_launch_records`：`config.json` / `command.txt` 用 `'x'` 模式写（**不可变**，已存在则静默跳过）；`run_attempts/{launch_id}.json`（schema `trellis.train-launch.v1`）记录 `command`、`source_sha256`（`train.py` 自身 sha256）、`requested_checkpoint`、`resolved_checkpoint`、`output_dir`、`load_dir`、`config`、`attempts`、`status`、`max_attempts`。
- `run_attempts`：`--auto_retry` 语义是**最大总尝试次数**（0 → 1，不是"额外重试"）。失败后重试时强制 `load_dir = output_dir`、`ckpt = latest`；若当前 output 没有 complete 检查点则拒绝重启（"refusing restart from 0 or replay of initial source"），并抛 `RuntimeError('Training failed; safe retry unavailable: ...')`；耗尽次数抛 `RuntimeError('Training failed: exhausted N maximum attempts')`。

### 5.5 配置命名规律

| 前缀 | 含义 |
|---|---|
| `ss_flow_*` | SS flow（`SparseStructureFlowModel`），数据集 `ImageConditionedSparseStructureLatentViews` |
| `slat_flow_img2shape_*` | 官方 shape SLat flow（图像→shape） |
| `slat_flow_imgshape2tex_*` | 官方 texture SLat flow（图像+shape→texture） |
| `shape_flow_*` | 本项目 shape SLat flow 配置 |
| `tex_flow_*` | 本项目 texture SLat flow 配置 |
| `*_lora_*` | 启用 `lora` 段（`rank` 8 或 16） |
| `*_dinov3_*` | 条件编码器为 `DinoV3FeatureExtractor`（`./dinov3`），`image_size` 512 或 1024 |
| `*_canon*` / `*_side*` / `*_alignedyz*` | SS 数据集的 latent 帧口径：`latent_frame='canonical'`（配 `canonical_latent_root`）或 `'view'`；`min_view_aspect` / `max_view_offaxis` 视角过滤 |
| `*_detail` | 更高分辨率/更细配置（`resolution=1024`、`real_view_prob=0.75`） |
| `*_v2` | 同一阶段的第二版配方 |
| `*_ft_*` / `*_ft1024` | 全量微调（无 `lora` 段） |
| 末尾数字（`20k`/`50k`/`_to_29000`/`_to_1000k`） | `max_steps` 口径 |

**LoRA 配置要点**（以 `configs/gen/ss_flow_lora_64_views_dinov3_512_50k.json` 为例，全文 107 行已核实）：
`models.denoiser.name=SparseStructureFlowModel`（`resolution=16`、`in_channels/out_channels=8`、`model_channels=1536`、`cond_channels=1024`、`num_blocks=30`、`num_heads=12`、`mlp_ratio=5.3334`、`pe_mode=rope`、`share_mod=true`、`initialization=scaled`、`qk_rms_norm/qk_rms_norm_cross=true`、`dtype=bfloat16`）；`pretrained.denoiser = checkpoints/ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors`；`lora = {enabled: true, models:["denoiser"], rank:8, alpha:8.0, dropout:0.0, targets:[5 个]}`；`trainer.args = {max_steps:50000, batch_size_per_gpu:4, batch_split:2, optimizer:{AdamW, lr:1e-4, weight_decay:0}, ema_rate:[0.9999], mix_precision_mode:'amp', mix_precision_dtype:'bfloat16', grad_clip:{AdaptiveGradClipper, max_norm:1.0, clip_percentile:95}, i_log:100, i_sample:5000, i_save:1000, p_uncond:0.1, t_schedule:{logitNormal, mean:1.0, std:1.0}, sigma_min:1e-5, image_cond_model:{DinoV3FeatureExtractor, ./dinov3, 512}, snapshot_sampler_params:{steps:12, guidance_strength:7.5, guidance_rescale:0.7, guidance_interval:[0.6,1.0], rescale_t:5.0}}`。

**注意差异**：`configs/gen/ss_flow_lora_64_views_train.json`（早期 run）用 `DinoV2LocalFeatureExtractor`（`/data/lzx/ReconViaGen/checkpoints_dinov2`，`image_size=518`，`crop_scale` 默认 1.2），与推理管线的 DINOv3@512 特征空间不一致 —— 这是后续所有 `_dinov3_*` 配置存在的原因（`scripts/run_ss_lora_dinov3_512_50k.sh` 头部注释记录：SS occupancy IoU 0.525 → 0.054）。

---

## 6. 推理与评测链路

### 6.1 `scripts/infer_finetuned.py`

**职责**：加载官方 pipeline（`--checkpoints_dir`，默认 `./checkpoints`），按需替换为微调权重，批量推理并写**原子运行清单**。

关键函数：
- `load_finetuned_weights(pipeline, ...)`：先对**所有**请求路径做 `require_file` 预检（"Validate every requested path before mutating any model"），再依次加载。
  - full 权重（`torch.load(..., weights_only=True)` + `checked_load`）：`vae_decoder` → `pipeline.models['shape_slat_decoder']`、`flow_512` → `shape_slat_flow_model_512`、`flow_1024` → `shape_slat_flow_model_1024`。
  - LoRA adapter（`apply_lora_to_linear_layers` + `validate_lora` + `checked_load`）：`sparse_structure_lora` → `sparse_structure_flow_model`、`flow_512_lora` → `shape_slat_flow_model_512`、`flow_1024_lora` → `shape_slat_flow_model_1024`、`tex_slat_lora` → `tex_slat_flow_model_1024`（CLI 名为 `--tex_flow_lora`）。加载前 `state.pop('rope_phases')`。
- `infer_single(image_path, pipeline, output_dir, seed, envmap, max_faces=16777216, no_video, geometry_only, fast_glb, timings)`：`pipeline.run(image, seed=seed)[0]`（用 `pipeline_timing` 包住 `sample_sparse_structure`/`sample_shape_slat`/`sample_shape_slat_cascade`/`sample_tex_slat` 打点）→ `mesh.simplify(max_faces)` → 导出。**先导 GLB 再渲染视频**（"GLB 是主要产物，视频渲染耗时长且容易被中断"）。
  - `--geometry_only` → `{name}.ply`（顶点先过 `glb_coordinates`，无 UV）；
  - `--fast_glb` → 无纹理 `{name}.glb`；
  - 默认 → `o_voxel.postprocess.to_glb(...)` + `glb.export()`，再 `render_utils.make_pbr_vis_frames(render_utils.render_video(mesh, envmap=envmap))` 出 `{name}.mp4`（fps=15）。
- `main()`：

真实 CLI 参数（`argparse`，已逐一核对）：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--image` / `--input_dir` | 必需（互斥组） | 单图 / 目录（png/jpg/jpeg/webp） |
| `--output` | `./inference_output` | 输出目录 |
| `--checkpoints_dir` | `./checkpoints` | 预训练管线目录 |
| `--vae_decoder` / `--flow_512` / `--flow_1024` | None | full 权重显式路径 |
| `--flow_512_lora` / `--flow_1024_lora` / `--tex_flow_lora` / `--sparse_structure_lora` | None | LoRA 权重显式路径 |
| `--lora_rank` / `--lora_alpha` / `--lora_dropout` | 8 / 8.0 / 0.0 | 全局 LoRA 超参 |
| `--ss_lora_rank` / `--ss_lora_alpha` / `--shape_lora_*` / `--tex_lora_*` | None | 分阶段覆写 |
| `--vae_only` | False | 只换 VAE decoder，不自动探测 512 flow（显式 `--flow_1024` 仍生效） |
| `--pretrained_only` | False | 纯官方权重（不能与任何显式微调权重组合） |
| `--pipeline_type` | `1024_cascade` | `512`/`1024`/`1024_cascade`/`1536_cascade` |
| `--seed` | 42 | — |
| `--ss_swap_yz` | False | SS 输出 y/z 全局交换 |
| `--ss_canonicalize` | False | 按样本把 SS 最长轴规范到轴 1（优先于 `--ss_swap_yz`） |
| `--ss_cond_model` | `pipeline` | `pipeline`=共用管线 DINOv3；`dinov2`=DINOv2Local@518 + crop 1.2 |
| `--dinov2_path` | `/data/lzx/ReconViaGen/checkpoints_dinov2` | — |
| `--no_video` | False | 跳过视频 |
| `--geometry_only` | False | 导出 PLY（与 `--fast_glb` 互斥） |
| `--fast_glb` | False | 无纹理 GLB（跳过 remesh/UV） |
| `--max_faces` | 16777216 | `simplify` 上限 |
| `--image_timeout_seconds` | 0 | 每图 SIGALRM 超时（需 Unix；CUDA 扩展可能延迟投递） |

**校验顺序**（全部先于管线构建/显存分配）：超时有限且非负 → `geometry_only`/`fast_glb` 互斥 → `pretrained_only` 组合冲突 → `vae_only` 组合冲突 → 各 rank>0 → 各 alpha 有限且>0 → dropout ∈ [0,1) → `max_faces>0` → 显式路径 `require_file` → `discover_images`（去重 stem、非空）→ 创建 output → **已存在 `run_manifest.json` 直接 `FileExistsError`**。

**自动探测**（仅在未显式给出且非 `pretrained_only` 时）：
- `results/latest_vae/ckpts/`：优先 `decoder_latest.pt`，否则 `decoder_step*.pt`（降序），再 `decoder_ema*.pt`（降序）。
- `results/latest_flow/ckpts/`：优先 `denoiser_latest.pt`，否则 `denoiser_ema*.pt`，再 `denoiser_step*.pt`。**不会自动设置 `--flow_1024`**（注释：把 512 权重移植进 1024 会导致 1536_cascade 精修阶段表面破碎）；且传入 `--flow_512_lora` 时跳过该探测（避免旧全量 512 检查点覆盖官方基线）。

**产物：`run_manifest.json`**（schema `trellis.inference-run.v2`，`write_manifest` 用 `.tmp` + `os.replace` 原子写）。字段：
`flags`（完整 CLI 快照）、`seed`、`pipeline_type`、`export`（来自 `export_metadata()`：`coordinates='GLB (x,z,-y), proper rotation'`、`mode ∈ {formal_remesh_textured, raw_no_remesh}`、`formal_acceptance_eligible`）、`explicit_weights`、`resolved_weights`、`base_weights`（`local_pipeline_weights()` 解析 `pipeline.json` 的每组件权重并记 sha256）、`automatic_weights`、`official_components_retained`、`load_timings`、`actual_base_sources`（每模型 `config`/`weights` 的 path+size+sha256 + `computed_buffers_regenerated`）、`loaded_components`、`actual_coordinate_flags`、`acceptance_flags_eligible`、`samplers`、`lora`（`targets`、`expected_layers:150`、各阶段 rank/alpha、`alpha_verification` 说明 alpha 无法从历史检查点推断）、`images[]`（每图 `input`/`input_artifact`/`status`/`outputs`(path,size,sha256)/`timings`/`started_utc`/`finished_utc`/`error`）。

**失败语义**：`run_batch` 逐图 try/except，单图失败不中断；只要有失败或输入为空，`status='failed'`，`main()` 返回 1。顶层异常捕获后把 `status='failed'` + `run_error` 写回 manifest 并 `sys.exit(1)`。

### 6.2 `scripts/utils/inference_contract.py`

**职责**：无 torch/CUDA/模型依赖的轻量契约层（"no torch/CUDA/model imports"），供推理与测试共用。

| 函数 / 类 | 作用 |
|---|---|
| `TARGETS` | 5 个 LoRA 目标关键词（与 `train.py` 默认一致） |
| `require_file(path)` | 解析并断言是文件，否则 `FileNotFoundError('explicit file unavailable')` |
| `HashCache.record(path)` | 按 `(path, size, mtime_ns)` 缓存，分块 sha256，返回 `{path, size_bytes, sha256}` |
| `checked_load(model, state, allow_rope=True, pop_rope=False)` | 先比键集合（`rope_phases` 可豁免）再比 shape，`load_state_dict(strict=False)` 后再次核对，最后 `model.eval()` |
| `validate_lora(...)` | §5.2 |
| `glb_coordinates(vertices)` | `(x,y,z) → (x,z,-y)`，proper rotation |
| `export_metadata(geometry_only, fast_glb)` | 导出模式与正式验收资格 |
| `timed` / `pipeline_timing` | 阶段计时（`[STAGE START]`/`[STAGE END]`，可选 CUDA synchronize）；`pipeline_timing` 临时包裹 4 个采样方法并在退出时**精确还原**（自有属性删除、原本属性恢复） |
| `image_deadline(seconds)` | `SIGALRM` 单图上限；非 Unix 抛 `RuntimeError`；`<0`/非有限抛 `ValueError` |
| `write_manifest(path, data)` | 原子写 JSON（`.tmp` + `os.replace`） |
| `run_batch(paths, runner, manifest, manifest_path, timeout=0, hashes=None)` | 逐图执行，写 `images[]` 与 `status`，返回 0/1 |
| `discover_images(image, input_dir)` | 单图或目录扫描；重复 stem 直接拒绝（"would overwrite outputs"） |
| `local_pipeline_weights(checkpoints_dir, hashes)` | 读 `pipeline.json`，对每个组件尝试 `{value}.safetensors`/`.pt`，缺失则标 `remote_or_cache_unresolved` |

### 6.3 `scripts/utils/evaluate_fixed_frame.py`

**职责**：**CPU** 固定世界坐标帧诊断，明确声明"never an automatic geometry fidelity gate"。

CLI：`--root`（内置历史记录集）或 `--manifest`（二选一）、`--output`（必需，**已存在则 `FileExistsError`**）、`--reference-angle`（默认 10）、`--manual-reviews`。

核心函数：`world_mesh(scene)`（应用每个 instance 的完整世界变换，含重复实例，再 concatenate）、`mesh_metrics(mesh, reference_angle=10)`、`log_provenance`、`UnsafeRunContract`、`hash_record_metadata`、`verify_native_run_manifest`、`verify_run_provenance`、`evaluate_record`、`existing_manifest`、`run`。

`mesh_metrics` 输出：顶点/面数、退化面数与容差、AABB 与 `extents_xyz`、`longest_aabb_axis`、表面积、**面积加权面心协方差**（proxy）与 **精确均匀三角面表面积分协方差**（primary，含面内协方差项，故对共面细分不变）、PCA 特征值/主轴/`major_to_z_offangle_degrees`、`axis_ambiguous`（`gap < 1.5` 时主轴不可信）、以及 `z_over_y`/`z_over_x`/`x_over_y` 等比例。`orientation_reference_status = 'pass'` 条件：**最长 AABB 轴为 z 且 surface PCA 主轴与 z 夹角 ≤ reference_angle**。

`verify_run_provenance` 接受两种 schema：
- `trellis.inference-run.v2` → `verify_native_run_manifest`：要求 `status='success'`、`seed==42`、`pipeline_type=='1024'`、`actual_coordinate_flags` 中 `ss_canonicalize`/`ss_swap_yz` 均为 **False**、CLI flags 中 `ss_canonicalize`/`ss_swap_yz`/`fast_glb`/`geometry_only` 均为 **False**、`acceptance_flags_eligible is True`、`export.mode=='formal_remesh_textured'`、`formal_acceptance_eligible is True`、必需 base source（`sparse_structure_flow_model`/`sparse_structure_decoder`/`shape_slat_flow_model_1024`/`shape_slat_decoder`）与 resolved weights 的 path/size/sha256 记录完整、所有 resolved override 都在 `loaded_components` 中、成功输出图记录唯一且 GLB sha256/size 与实测一致、输入图 sha256 匹配。
- `trellis.fixed-frame-run-evidence.v1`（历史显式证据）：要求 `export_contract=='formal_standard_glb'`、seed/pipeline 一致、4 个 flag 均为 False、`load_validation=='strict_validated'`、loaded models 含 `sparse_structure`/`shape_1024`/`vae_decoder` 三角色且文件 sha256 实测一致、输入 sha256 一致。

不安全条件抛 `UnsafeRunContract` → `provenance_status='fail'` 并使 `acceptance_status='fail'`；其他异常 → `'provenance_unverified'`。`command_file` 中若出现 `--fast_glb`/`--geometry_only`/`--ss_canonicalize`/`--ss_yz_swap` 与固定帧契约矛盾，直接判 fail。

**产物**：`{output}/input_manifest.json`（使用的输入记录）与 `{output}/report.json`（schema `trellis.fixed-frame-report.v2`，含 `results[]`、`all_records_acceptance ∈ {pass, fail, unverified}`、`limitations` 明示"AABB/PCA/projected bbox 是 proxy，不是几何保真度；缺人工复核即为 unverified"）。`acceptance_status='pass'` 需同时满足：orientation pass + 人工几何复核 pass（`reviewer`/`evidence`/`rationale` 三者齐备）+ provenance verified。

---

## 7. 数据工具链 `data_toolkit/`

### 7.1 官方 7 步流程（`data_toolkit/README.md`，已核实）

| 步骤 | 脚本 | 关键参数 |
|---|---|---|
| 1 环境 | `setup.sh`（`. ./data_toolkit/setup.sh`）、`blender_script/install_pillow.py` | — |
| 2 元数据 | `build_metadata.py <SUBSET> --root <ROOT> [--source <SOURCE>]` | 另支持 `--download_root --thumbnail_root --render_cond_root --mesh_dump_root --pbr_dump_root --dual_grid_root --pbr_voxel_root --ss_latent_root --shape_latent_root --pbr_latent_root --field --from_file --from_merged_records --record_start --rebuild`；子集：`ObjaverseXL`/`ABO`/`HSSD`/`TexVerse`（训练）、`SketchfabPicked`/`Toys4k`（测试） |
| 3 下载 | `download.py <SUBSET> --root <ROOT>` | `--rank/--world_size`、`--filter_low_aesthetic_score`、`--check_only`、`--instances` |
| 4 dump 资产 | `dump_mesh.py`、`dump_pbr.py`（CPU，走 Blender）、`asset_stats.py` | `--mesh_dump_root`、`--pbr_dump_root`、`--max_workers` |
| 5 转 O-Voxel | `dual_grid.py`、`voxelize_pbr.py` | `--resolution 256,512,1024`（默认 256） |
| 6 编码 latent | `encode_shape_latent.py`、`encode_pbr_latent.py` → `build_metadata.py` → `encode_ss_latent.py --shape_latent_name <NAME> --resolution 64` | `--resolution` 默认 1024（输入 O-Voxel 分辨率）；`--enc_pretrained/--model_root/--enc_model/--ckpt` |
| 7 渲染图像条件 | `render_cond.py <SUBSET> --root <ROOT>` | `--num_cond_views`（默认 16）、`--render_cond_root` |

**Blender 侧脚本**（`data_toolkit/blender_script/`，被上面各步通过 `_install_blender()` 调用）：`dump_mesh.py`（`init_scene`/`load_object`/`delete_invisible_objects`/`scene_bbox`/`normalize_scene`）、`dump_pbr.py`（额外 `extract_image`/`try_extract_image`/`try_extract_factor`/`try_extract_image_with_factor`）、`render_cond.py`（`init_render`/`init_camera`/`init_uniform_lighting`/`init_random_lighting`/`convert_to_meshes`/`triangulate_meshes`/`get_transform_matrix`）、`render_view.py`（`init_white_lighting` 等）。另有 `io_scene_usdz.zip`（USDZ 导入插件包）。

### 7.2 本项目新增的脚本与其关系

```text
资产输入 (GLB/OBJ/blend)
   │
   ├─ custom_prepare_data.py --data_dir --output_dir --resolutions --flat --conditions-only
   │     一次性准备：生成 dual-grid / 形状 dump / 条件渲染
   │
   ├─ 规范帧 (canonical) 目标
   │     canonical_frame.py                  ← 坐标约定的唯一权威定义（§3.2）
   │     build_canon_canonical_latents.py    (--src --output_name --workers)
   │     build_ss_canonical_contract.py      (--root --output --max-angle)
   │
   └─ 渲染条件
         render_cond.py (顶层包装) → blender_script/render_cond.py
```

逐视角体素/latent 生成脚本（`generate_ss_view_voxels*.py`、`encode_ss_view_latents*.py`、`generate_ss_view_from_shape_coords.py`）已删除；现行训练使用每对象唯一的 canonical SS latent，不再走逐视角支撑。

**关系与产物落点**（与 `training_data/multiview_1186/` 的目录一一对应，已实测该目录内容）：
`mesh_dumps` / `pbr_dumps` → `dual_grid_512`、`dual_grid_1024`（shape 输入）→ `shape_latents/*`；`pbr_dumps` → `pbr_voxels_1024` → `pbr_latents/*`；`ss_view_voxels_canon_v4` → `ss_latents_views_canon`（逐视角 SS 监督，供 `ImageConditionedSparseStructureLatentViews`）；`ss_latents_canon`（每对象唯一规范帧 latent，`latent_frame='canonical'` 时用）；`renders_cond`（原始视角渲染）与 `renders_cond_t`（带 `transforms.json`、`_view_aspect.json`、`_view_offaxis.json` 的契约化渲染，canonical 契约与 `max_view_offaxis` 过滤依赖它们）。

**契约构建产物**：`build_ss_canonical_contract.py --root <训练数据根> --output <contract 目录> --max-angle <度>` 会生成 `canonical/`、`COMPLETE.json`、以及需人工审批的 `TRAINING_APPROVAL.json` + `coordinate_evidence`/`mask_evidence`；`trellis2/datasets/canonical_contract.py::load_canonical_contract` 按 §4.4 逐项校验（默认拒绝 staged 数据）。

> 未核实项：`custom_prepare_data.py`、`build_canon_canonical_latents.py` 的内部实现细节仅确认函数与参数名，未逐行审阅。

---

## 8. 测试 `tests/`

共 **10 个** `unittest` 文件（无 `pytest.ini`/`conftest.py`/`tests/__init__.py`；用 `unittest discover` 运行）。测试以 CPU-only 方式执行（历史命令带 `CUDA_VISIBLE_DEVICES=''`）。

| 文件 | 覆盖点 |
|---|---|
| `test_canonical_frame.py` | 坐标帧：标记 (up/depth/length) 与实际 Blender 一致、所有视角变换为 proper rotation 且保持 up、native 长 x + yaw90 → 内部 y、支撑体转换是"网格双射"而非特征变换、非法支撑/顶点必须报错 |
| `test_checkpoint_contract.py` | 检查点契约：原子+完整+hash+sync、保存失败不产生 marker、首文件后失败保持 partial 且无 complete、拒绝覆盖 complete/partial、坏 marker 必须抛错不得静默取 latest、仅 misc 不安全、同尺寸篡改被 hash 发现、size 缺失/不符、legacy 文件不算 complete、CPU 快照不别名、RNG 往返（python/numpy/torch）、非 rank0 不收 rank0 RNG、legacy RNG 告警、精确 fp32 adapter 与 complex buffer 保留、非 LoRA 模型不被强制 fp32、`run()` 末步补存不重复、snapshot 异常传播、final run 分支 skip/enabled |
| `test_inference_contract.py` | 推理契约：rope 缺失允许且切 eval、坏 missing 在变更前拒绝、unexpected/shape 拒绝、`validate_lora` 150 层正例/缺对/层数错/rank·alpha 错、缺文件、hash 只算一次、旋转输入不变且行列式、raw 不等于正式、批量部分失败 → manifest 非零、空批量非零、空输出失败、discover 预检、`pipeline_timing` 精确还原方法、超时中断、非法超时、脚本 main 预检先于管线、`geometry_only` 旋转后不算双重旋转、显式权重失败先于模型、缺图先于模型、空目录先于模型、main 部分失败 manifest 标记、`pretrained`+LoRA 冲突 |
| `test_lora_idempotence.py` | LoRA 幂等：重复注入保持身份键值输出、内部关键字绝不包裹已有 adapter、新未包裹目标仍可注入、嵌套 `Sequential` 重复、已包裹根层不变 |
| `test_pretrained_loading.py` | 预训练加载：精确权重 + RoPE 重建、缺权重失败、多余权重失败、不完整本地路径不触网、本地组件缺失不回退、损坏本地组件绝不回退、跨仓库组件直载 |
| `test_ss_canonical_contract.py` | SS 规范契约：pose 接受 length1 而不是 depth、非法 coords、目标 dtype/shape/有限性、帧遍历、已存在输出拒绝、NaN 角度拒绝 |
| `test_ss_canonical_integration.py` | 集成：只允许 native 与已接受 target、损坏 target 失败且无 complete、负缓存不得被批准 |
| `test_ss_contract_loading.py` | staged 默认拒绝、显式诊断可打开、manifest 篡改拒绝、源图篡改拒绝、缺 complete 拒绝、仅有 approval 布尔而无证据拒绝、root 不匹配拒绝、数据集契约视角与 staged 守卫 |
| `test_train_launch_contract.py` | 启动契约：空 fresh latest 显式 none、latest 需 marker 而非 misc、config 历史阻止 none/空 latest、latest 选 complete 而非最高 misc、选中 complete 被篡改则失败、legacy 数字告警且要求整组、数字 marker 篡改不回落 legacy、distinct load 续到新 output、外部续训不得混已有 output、外部空 latest 不算 fresh、保留历史且 attempt 日志唯一、首次 none 失败绝不从 0 重启、外部初始续训无提交即拒绝重放、重试把 none 改为本地已提交 latest、达最大次数不再多报重试、0 重试=1 次、失败子进程非零、attempt 记录持久化、真实原子 latest 忽略未提交 misc、真实原子篡改与坏 marker fail-closed、fresh 守卫 AST 先于历史写、实际 AST 启动终失败传播并记录、守卫失败不改历史、`train.py` 的 `find_ckpt` 委托、pretrained 的 rope 保留与坏键失败 |
| `test_fixed_frame_acceptance.py` | 固定帧验收：对齐盒朝向且不改输入、倾斜不被长 z 掩盖、场景世界变换与重复实例、空/退化拒绝、非有限拒绝、含有效面的退化计数、缺失算 fail 而非 skip、无人工复核即 unverified、fast 导出与轴修正被拒、已校验 manifest 必需且 hash 变更被拒、native v2 manifest 的 hash/flags/人工复核、表面积单三角拆分不变性、矩形重细分解析矩、旋转 float32 盒细分不变性、报告保留缺失第四项且拒绝覆盖 |

---

## 9. 常用命令速查

> 全部从仓库内真实脚本与 `argparse` 定义推导。`PY` 取仓库脚本使用的解释器：`/home/test/miniconda3/envs/trellis2/bin/python`。**不要使用 GPU 0–3**。

### 9.1 训练（LoRA）

```bash
cd /data/lzx/TRELLIS.2

# Stage 1 SS LoRA（DINOv3@512，50k）—— 直接跑脚本，GPU 作为第 1 个参数
bash scripts/run_ss_lora_dinov3_512_50k.sh 4

# Stage 2 shape 512 LoRA（脚本内固定 CUDA_VISIBLE_DEVICES=4）
bash scripts/run_stage2_512_lora.sh

# Stage 2 shape 1024 LoRA v2（200k，脚本内固定 GPU4）
bash scripts/run_stage2_lora_v2_resume_200k.sh

# Stage 3 texture 1024 LoRA（GPU 作为第 1 个参数，默认 6）
bash scripts/run_tex_flow_lora.sh 6
```

上述脚本展开后的等价形式（以 SS 为例）：

```bash
CUDA_VISIBLE_DEVICES=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
$PY -u train.py \
  --config configs/gen/ss_flow_lora_64_views_dinov3_512_50k.json \
  --output_dir ./results/multiview_1186_ss_flow_lora_64_views_dinov3_512 \
  --ckpt none --num_gpus 1 --auto_retry 3 \
  --data_dir '{"train_data":{"base":"./training_data/multiview_1186","ss_view_latent":"./training_data/multiview_1186/ss_latents_views_mesh_t1","render_cond":"./training_data/multiview_1186/renders_cond"}}'
```

续训（把 `--load_dir` 指向同一 output 并 `--ckpt latest`）：

```bash
$PY -u train.py \
  --config configs/gen/shape_flow_lora_1024_v2_train.json \
  --output_dir ./results/multiview_1186_flow_1024_lora_v2 \
  --load_dir  ./results/multiview_1186_flow_1024_lora_v2 \
  --ckpt latest --num_gpus 1 --auto_retry 3 \
  --data_dir '{"train_data":{"base":"./training_data/multiview_1186","shape_latent":"./training_data/multiview_1186/shape_latents/shape_enc_next_dc_f16c32_fp16_1024","render_cond":"./training_data/multiview_1186/renders_cond"}}'
```

只构建不训练（配置自检）：追加 `--tryrun`。

### 9.2 微调推理

单图（SS 33k EMA + shape 1024 20k EMA + 官方 decoder）：

```bash
CUDA_VISIBLE_DEVICES=7 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
$PY -u scripts/infer_finetuned.py \
  --image   ./data_test/001.png \
  --output  ./results_infer/my_run_001 \
  --pipeline_type 1024 --vae_decoder ./checkpoints/ckpts/shape_dec_next_dc_f16c32_fp16.pt --vae_only \
  --sparse_structure_lora ./results/multiview_1186_ss_flow_lora_64_views/ckpts/denoiser_ema0.9999_step0033000.pt \
  --ss_lora_rank 8 --ss_lora_alpha 8 \
  --flow_1024_lora ./results/multiview_1186_flow_1024_lora_v2/ckpts/denoiser_ema0.9999_step0020000.pt \
  --shape_lora_rank 8 --shape_lora_alpha 8 \
  --tex_flow_lora ./results/multiview_1186_tex_flow_lora_1024/ckpts/denoiser_ema0.9999_step0100000.pt \
  --tex_lora_rank 8 --tex_lora_alpha 8 \
  --no_video
```

批量目录（真实脚本 `scripts/infer_ss33k_shape20k_texN_gpu.sh <tex_step> <gpu>`）：

```bash
bash scripts/infer_ss33k_shape20k_texN_gpu.sh 100000 5
```

SS 步数扫描（`scripts/infer_ss_dinov3_sweep_gpu.sh <ss_step> <gpu>`）：

```bash
bash scripts/infer_ss_dinov3_sweep_gpu.sh 33000 5
```

纯官方基线：

```bash
$PY scripts/infer_finetuned.py --image ./data_test/001.png --output ./results_infer/ctrl_base_x --pretrained_only --pipeline_type 1024 --no_video
```

快速几何对比（无 UV，出 PLY）：

```bash
$PY scripts/infer_finetuned.py --image ./data_test/001.png --output ./results_infer/geom_x --geometry_only --pipeline_type 1024
```

> 注意：`--output` 目录若已存在 `run_manifest.json`，脚本会直接报错，必须换新目录。

### 9.3 评测（固定帧验收）

```bash
# 使用内置历史记录集（--root）
$PY scripts/utils/evaluate_fixed_frame.py --root /data/lzx/TRELLIS.2 --output ./diagnostics/my_fixed_frame_$(date +%Y%m%d) --reference-angle 10

# 或使用自定义 manifest
$PY scripts/utils/evaluate_fixed_frame.py --manifest my_records.json --output ./out --manual-reviews reviews.json
```

单文件 GLB 几何指标：

```bash
$PY scripts/utils/mesh_metrics.py path/to/001.glb --json        # 位置参数: glbs [--json]
$PY scripts/utils/validate_glb.py --root ./results_infer/my_run_001
```

### 9.4 跑测试

```bash
cd /data/lzx/TRELLIS.2
CUDA_VISIBLE_DEVICES='' $PY -m unittest discover -s tests -v
# 单个文件
CUDA_VISIBLE_DEVICES='' $PY -m unittest discover -s tests -p test_fixed_frame_acceptance.py -v
```

### 9.5 官方示例 / Demo

```bash
$PY run_trellis.2.py           # 固定 CUDA_VISIBLE_DEVICES='7'，读 ../single_side/*.png
$PY data_toolkit/build_metadata.py ObjaverseXL --source sketchfab --root datasets/ObjaverseXL_sketchfab
```

### 9.6 环境补丁

```bash
$PY scripts/utils/apply_patches.py            # 应用（含清理 __pycache__）
$PY scripts/utils/apply_patches.py --reverse  # 撤销
```

---

## 10. 代码阅读建议

**想理解"图像 → SS"**（按顺序读）：
1. `trellis2/pipelines/trellis2_image_to_3d.py` — `preprocess_image` → `get_cond` / `get_ss_cond` → `sample_sparse_structure`（含 `ss_res` 字典、`argwhere`、`ss_canonicalize` / `ss_swap_yz` 分支）。
2. `trellis2/modules/image_feature_extractor.py::DinoV3FeatureExtractor` — 条件特征怎么来的。
3. `trellis2/models/sparse_structure_flow.py::SparseStructureFlowModel` — 16³×8 的 DiT 与 `rope_phases`。
4. `trellis2/models/sparse_structure_vae.py::SparseStructureDecoder` — latent 如何变成占据。
5. `trellis2/pipelines/samplers/flow_euler.py` + `guidance_interval_mixin.py` — CFG 区间采样。
6. 训练侧口径：`trellis2/datasets/sparse_structure_latent_views.py`，以及 `scripts/infer_ss_dinov3_sweep_gpu.sh` 头部注释记录的条件空间不一致问题。

**想理解"训练"**：
1. `train.py`（入口与 LoRA 注入顺序）。
2. `trellis2/trainers/basic.py` — `__init__` → `load` / `save` / `run` / `run_step` / `snapshot` / `update_ema`。
3. `trellis2/trainers/flow_matching/flow_matching.py` 与 `sparse_flow_matching.py` — `training_losses` / `sample_t` / `get_cond`。
4. `trellis2/trainers/flow_matching/mixins/classifier_free_guidance.py` 与 `image_conditioned.py` — `p_uncond` 与条件注入。
5. `trellis2/utils/train_launch_contract.py` + `checkpoint_contract.py` — 恢复与提交语义。
6. `configs/gen/ss_flow_lora_64_views_dinov3_512_50k.json` — 一份完整可对照的配方。

**想理解"LoRA"**：
1. `trellis2/modules/lora.py`（全文 74 行，`LoRALinear` + `apply_lora_to_linear_layers`）。
2. `train.py:80-97`（冻结 + 注入 + `replaced` 空则报错）。
3. `trellis2/modules/sparse/attention/modules.py`（`to_qkv`/`to_q`/`to_kv`/`to_out` 的定义，决定了参数形状）。
4. `trellis2/modules/sparse/transformer/modulated.py`（`self.self_attn` / `self.cross_attn` 名字来源，决定 150 = 30×5）。
5. `scripts/utils/inference_contract.py::validate_lora` + `scripts/infer_finetuned.py::load_finetuned_weights`（推理侧严格校验与加载顺序）。
6. `tests/test_lora_idempotence.py`（幂等要求）。

---

## 11. 重要契约与坑

1. **检查点提交只认 complete marker。** `ckpts/complete_step%07d.json` 是唯一可见性判定点；`misc_step*.pt` 单独存在**不是**提交。`--ckpt latest` 只选 complete 中最大 step；无 complete 但有历史 → 直接报错，**不会**退化成"最新的 misc"。中途失败的文件与 `.reservation` 都**故意保留**作为证据，且**不允许原地覆盖**——失败后必须换新的 run 目录。
2. **`--auto_retry` 是最大尝试次数**（0 → 1 次），不是"额外重试 N 次"。失败后只能从**本 output** 已提交的 checkpoint 续跑；没有提交就拒绝重启（防止重放不确定的优化器轨迹）。
3. **外部断点续训必须换全新 output 目录**（`--load_dir != --output_dir` 且 output 已有历史 → 报错）。`config.json` / `command.txt` 一旦写入不可变（`'x'` 模式）。
4. **显式权重路径必须真实存在。** 推理侧所有 `--vae_decoder/--flow_512/--flow_1024/--*_lora` 都先 `require_file` 再动模型；`Pipeline.from_pretrained` / `models.from_pretrained` 对本地组件缺失、架构不符一律抛错，**绝不静默回退到其他权重或联网**。
5. **LoRA 严格校验（150 层）。** 目标 5 个 × 30 blocks = 150；推理前 `validate_lora` 会核对层数、key 集合、target 合法性、运行时 rank/alpha、down/up 形状。`alpha` **不在**历史 state_dict 里，manifest 明确记录"CLI 值仅供参考，未从检查点推断"。LoRA 注入**幂等**（不会二次包装已包装层）。
6. **`rope_phases` 永远重新生成、从不从检查点加载。** SS flow 的 `rope_phases` 是 16³ × 64 = 262,144 元素的 complex buffer，加载时一律 `pop`，缺失键豁免；`cast_model_preserving_lora_and_complex` 会保住 complex buffer 与 LoRA 的 fp32 精确值及 Parameter 身份（AMP master params 依赖身份）。
7. **坐标帧是硬契约。** 内部帧 → 正式 GLB = **(x, z, -y)**（`glb_coordinates` / `internal_to_formal_glb`）；raw 轴序 **(depth, length, up)**，raw 长度槽为 y；native glTF Y-up → 内部 = `v[:,[0,2,1]]*[1,-1,1]` 再绕 yaw（det = +1 的 proper rotation）。**固定的评测只接受 formal GLB**：`evaluate_fixed_frame.py` 会拒绝 `--fast_glb` / `--geometry_only` / `--ss_canonicalize` / `--ss_swap_yz`，并要求 seed=42、pipeline_type=1024。
8. **`--ss_swap_yz` / `--ss_canonicalize` 是修正手段，不是默认行为。** 二者都会把 `acceptance_flags_eligible` 置 False（不能在验收里用）；`ss_canonicalize` 优先于 `ss_swap_yz`。鱼与熊掌：本地 SS 监督缺 y/z 对齐，用本地 SS LoRA 时朝向依赖这些开关。
9. **自动权重探测有陷阱。** `results/latest_flow/ckpts/` 当前只有 `denoiser_latest.pt`（实测）；`results/latest_vae/ckpts/` 预期有 `decoder_latest.pt`。已显式传 `--flow_512_lora` 时**跳过** `latest_flow` 探测（否则旧全量 512 检查点会覆盖官方基线）；**永不自动设置 `--flow_1024`**（512 权重挪进 1024 会破坏 1536_cascade 精修）。
10. **输出目录必须新建。** `--output` 内已有 `run_manifest.json` → `FileExistsError`；评测 `--output` 已存在 → `FileExistsError`。这是刻意的防覆盖设计。
11. **`--checkpoints_dir` 的 `pipeline.json` 含越树绝对路径。** 实测 `checkpoints/pipeline.json` 的 `image_cond_model.model_name = /home/lyx/lzx/TRELLIS.2/dinov3`、`rembg_model.model_name = /home/lyx/lzx/TRELLIS.2/rmbg`。`/home/lyx/lzx` 实为指向 `/data/lzx` 的软链接，故等价于项目内 `dinov3/` 与 `rmbg/`；但换机器/换用户名会失效（发布包 `trellis2_finetune_release` 里改成了相对路径 `./dinov3`、`./rmbg`）。
12. **GPU 使用限制。** 项目文档反复声明：**GPU 0–3 被其它任务占用（各约 134 GB），不要使用**；可用范围为 4–7，且 GPU4–7 常有 ComfyUI 常驻（约 528 MiB、0% 利用率），**不要终止它**。`train.py` 在 `num_gpus < 1` 时直接拒绝启动（要求显式指定可用 GPU）。`run_trellis.2.py` 与 `infer_finetuned.py` 的默认 `CUDA_VISIBLE_DEVICES` 是 `'7'`；`scripts/*.sh` 里硬编码的是 4/5/6/7。
13. **`nju/` 只读。** 561 G，TRELLIS.1 相关代码与检查点（`3d_gen_online`、`3dgen_train`、`hy3d`、`trellis-lora`）。项目规则：只读、不改、不删。同理：不 `git reset`、不 `git checkout`、不提交（除非用户明确要求）；`results` 与 `training_data` 是指向 `/data/TRELLIS.2/` 的软链接，别当成独立目录移动。
14. **`scripts/utils/apply_patches.py` 会改写仓库内源码。** 它就地修改 3 处：① `trellis2/trainers/flow_matching/mixins/image_conditioned.py` 与 `trellis2/modules/image_feature_extractor.py` 的 `self.model.layer` → `self.model.model.layer`（transformers ≥ 5.14 兼容）；② `trellis2/trainers/flow_matching/sparse_flow_matching.py` 的 `run_snapshot` 加 `num_samples = min(num_samples, data['x_0'].shape[0])`（修 SparseTensor 空切片）；③ `trellis2/trainers/vae/shape_vae.py` 的 `run_snapshot` DataLoader（`batch_size=num_samples`、`num_workers=0`）。**改代码前先确认这些补丁是否已应用**，`--reverse` 可撤销。
15. **"latent 代理"不等于几何验收。** `snapshot_allowed(dataset)` 尊重数据集上的 `skip_snapshot`；SS 数据集默认 `skip_snapshot=True`（8 通道 3D latent 不适合通用可视化，且 trainer 的 `snapshot()` 未包 try，失败会终止训练）。`evaluate_fixed_frame.py` 也明确声明 AABB/PCA 是 proxy，缺人工复核即为 `unverified`。
16. **`trellis2_finetune_release/` 是发布副本，不是研究工作区。** 含 `scripts/infer.sh`、`merge_lora_into_base.py`、独立 `checkpoints/pipeline.json`（`default_pipeline_type='1024'`，条件模型用相对路径）。`FULL_AUDIT_20261013.md` 记录：发布的 `infer.sh` 默认用 GPU0（不适合本机）、且其例子在已合并的 ft33k 基座上再注入同一 LoRA 有**重复适配**风险；`merge` 脚本示例用 ft33k 当 base 与帮助文本矛盾。**不要把发布包的成功声明当作当前研究验收。**

---

### 附：本文档核实方式

- 目录与文件存在性：`ls`、`find`，并对本文列出的 **140+ 条路径逐条 `[ -e ]` 检查，缺失数 0**。
- 体积：`du -sh`（`results`/`training_data` 为其符号链接目标的真实体积）。
- 类名/函数名/参数名：读取仓库源码副本（`tar` 打包后下载，排除 `__pycache__`，未含任何数据文件）。
- LoRA 规模：用 `~/miniconda3/envs/trellis2/bin/python` + `torch.load(..., weights_only=True)` 读取真实检查点 `results/multiview_1186_ss_flow_lora_64_views/ckpts/denoiser_step0033000.pt` 统计（150 层 / 300 张量 / 4,669,440 可训练 / 1,296,849,416 基座，与 `HANDOFF_NEXT_20261012.md` 记录一致）。
- 坐标约定：`data_toolkit/canonical_frame.py` 与 `scripts/utils/inference_contract.py` 源码。
- **未核实 / 保守处理**：`trellis2/modules/spatial.py` 与 `sparse/spatial/*` 的逐算子下标推导；`scripts/utils/` 下一次性工具的算法细节（仅确认函数名与 CLI 参数）；`data_toolkit/custom_prepare_data.py` 与 `build_canon_canonical_latents.py` 的内部实现；`diagnostics/` 48 G 非 `.py`/`.md` 证据的具体内容；`assets/app` 图标资源清单。
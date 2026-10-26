# LoRA 方案 README（中国铁路动车组 · 当前参考方案）

> 本文件描述**当前正在使用的**那一套 LoRA 方案：数据池 pool1302 → 三条 LoRA 训练线 → 1024 直采推理 → 评测口径。
> 用途是**系统性梳理与检查 pipeline**：每一段都给出「输入 / 输出 / 关键代码位置 / 校验点」，末尾有可勾选的自检清单和已知坑。
> 上位文档：`PROJECT_CODE_GUIDE.md`（代码架构全貌）、`CURRENT_STATE.md`（当前状态与待办）。本文件不重复它们，只把「当前这条线」串起来。
>
> 核对时间：2026-10-26。所有路径、步数、超参、行号均来自**当前工作副本 `/data/lzx/TRELLIS.2` 的实测**，不是交接文档的转述。凡未经实测的都标了「未实测」。

---

## 0. 一分钟速览

| 项 | 值 |
|---|---|
| 任务 | 单张侧视/多视角工程图 → canonical 坐标下的动车组网格（GLB） |
| 数据集 | `training_data/pool1302/`：1302 对象 = train **1146** + holdout **156** |
| LoRA 线 1 | Stage1 稀疏结构 SS flow，`SparseStructureFlowModel`（16³ 潜在 / 64³ 占据），30k EMA |
| LoRA 线 2 | Stage2 shape SLat flow **1024**，`ElasticSLatFlowModel`，30k EMA |
| LoRA 线 3 | Stage3 texture SLat flow 1024，`ElasticSLatFlowModel`（64 通道），30k EMA，**尚未验收** |
| LoRA 形态 | rank 8 / alpha 8 / dropout 0，挂 5 类 attention 投影，每模型 **150 层** |
| 参考产物 | `results_infer/001_three_arms_v1/ss30_shape30/001.glb` |
| 推理口径 | `pipeline_type='1024'`（**不是** `1024_cascade`）、seed 42、SS/shape/tex 各 12 步、CFG 7.5 |
| 权重 | SS `…canon_axis10/ckpts/denoiser_ema0.9999_step0030000.pt`、shape `…1024_v2/ckpts/denoiser_ema0.9999_step0030000.pt` |
| 硬约束 | 只用 GPU 6/7；不碰 `nju/`；不覆盖已有结果目录；SS 条件图恒 512 |

---

## 1. 方案总览

```
                     data_for_train/<object_id>/                      (源数据，77 GB)
                       ├─ mesh_textured_pbr.glb   → id = sha256(glb)[:16]
                       ├─ canonical_geometry.npz  → GT 网格（评估用）
                       └─ 24 张 *_cam_*.png       → 条件渲染源
                                  │
   ┌──────────────────────────────┴───────────────────────────────┐
   │ diagnostics/pool_package_v1/build_pool1302.py  （caches→select→emit→verify）
   │   把 diagnostics/pool_targets_v2 的诊断编码落成训练就绪布局
   └──────────────────────────────┬───────────────────────────────┘
                                  ▼
        training_data/pool1302/{train,holdout}/   ← 布局与旧 multiview_1186 同构
          ├─ renders_cond_t/<id>/000_cam_000.png … 023_cam_330.png  (24×512², RGBA) + transforms.json
          ├─ ss_latents_canon/<id>.npz      (key 'mean', (8,16,16,16) float32)      ← SS 目标
          ├─ ss_view_latents_index/<id>/INDEX.json                                   ← 视角索引（canonical_only）
          ├─ shape_latents/shape_enc_next_dc_f16c32_fp16_1024/<id>.npz  (feats+coords) ← shape 目标
          ├─ pbr_latents/tex_enc_next_dc_f16c32_fp16_1024/<id>.npz                    ← tex 目标
          └─ metadata.csv （sha256 / split / object_id / *_encoded 标记 …）
                                  │
        ┌───────────┬─────────────┴────────────┬──────────────────┐
        ▼           ▼                          ▼                  ▼
  Stage1 SS     Stage2 shape 1024        Stage3 tex 1024        （可选）诊断
  run_ss_…sh    run_shape_…v2.sh         run_tex_flow_…sh       diagnostics/*
        │           │                          │
        ▼           ▼                          ▼
 results/pool1302_ss_flow_lora_64_canon_axis10   results/pool1302_shape_flow_lora_1024_v2   results/pool1302_tex_flow_lora_1024
   ckpts/denoiser_ema0.9999_stepNNNNNNN.pt (每 2k/1k 一存)
        │           │                          │
        └───────────┴──────────┬───────────────┘
                               ▼
                checkpoints/pipeline.json （基础管线，8 个组件）
                               ▼
    trellis2/pipelines/trellis2_image_to_3d.py::run(pipeline_type='1024')
      ① preprocess_image → ② get_cond(512/1024) → ③ sample_sparse_structure(64³)
      → ④ sample_shape_slat → ⑤ sample_tex_slat → ⑥ decode_latent(+fill_holes)
                               ▼
       scripts/infer_finetuned.py::infer_single  → o_voxel.postprocess.to_glb
                               ▼
                     *.glb / *.ply / *.mp4（正式产物 = GLB）
```

---

## 2. 数据侧：pool1302

### 2.1 来源与构建

| 步骤 | 命令 | 产物 / 校验 |
|---|---|---|
| 诊断编码（已完成，54 GB） | 池 → `diagnostics/pool_targets_v2/<object_id>/{ss_mean.npz, shape_mean.npz, ss_support64.npz, dual_grid1024.npz, decoded_*}` | `manifest.json` + `COMPLETE.json`（哈希绑定，`assets=1302`） |
| 落成训练布局 | `python diagnostics/pool_package_v1/build_pool1302.py caches` → `select` → `emit` → `verify` | 只读池与 `pool_targets_v2`；**只写** `--out`（默认 `training_data/pool1302/`） |
| 划分 | `SPLIT_PROPOSAL_20261015.json`，脚本内钉死 `SPLIT_SHA256 = d5f5e627f282bc47fcb700edf18735d2d6bcf3ccd66278eafcf488aa4d4f1bfd` | train 1146 / holdout 156，两者不重叠且覆盖全池 |

### 2.2 布局与键（`trellis2/datasets` 直接消费，代码零改动）

| 目录 | 内容 | 被谁用 |
|---|---|---|
| `renders_cond_t/<id>/` | 24 张 512×512 RGBA + `transforms.json`（内嵌 `png_sha256` 可复核） | 三个数据集共用（条件图） |
| `ss_latents_canon/<id>.npz` | 键 `mean`，(8,16,16,16) float32；**每对象一个规范帧 latent** | Stage1（`latent_frame='canonical'`） |
| `ss_view_latents_index/<id>/INDEX.json` | `schema=trellis.pool1302-view-index.v1`，`canonical_only: true`，`selected_views`（例 `[0,12]`） | Stage1 的 `ss_view_latent` 根 |
| `shape_latents/shape_enc_next_dc_f16c32_fp16_1024/<id>.npz` | `feats` + `coords`（1024 口径） | Stage2 |
| `pbr_latents/tex_enc_next_dc_f16c32_fp16_1024/<id>.npz` | 纹理 latent，坐标与 shape latent **逐 token 相等** | Stage3 |
| `metadata.csv` | `sha256,…,object_id,split,shape_latent_encoded,num_views,…` | 统计/审计 |

体积（train）：`renders_cond_t` 2.7 G、`shape_latents` 678 M、`ss_latents_canon` 135 M、`pbr_latents` 125 M、`ss_view_latents_index` 9.1 M。

### 2.3 关键口径（容易踩）

- **id 规则**：`id = sha256(mesh_textured_pbr.glb)[:16]`，与 `records.jsonl`、`manifest.json` 的 `mesh_sha256[:16]` 一致。
- **渲染是复制不是软链**：为了让数据集自洽可归档，2.87 GB 冗余是有意为之。
- **`ss_latents_canon` 是每对象单 latent**：逐视角 SS latent **没有物化**（`generate_ss_view_voxels*` / `encode_ss_view_latents*` 已删除），`INDEX.json` 的 `note` 明确写了这点。
- **offaxis 定义**：`_view_offaxis.json` = 规范帧 64³ 支撑 PCA 长轴与渲染相机 `camera_right_world` 的夹角，`≤10°` 才算侧视（复现旧 `build_ss_canonical_contract.py::pose()` 口径，`MAX_VIEW_OFFAXIS = 10.0`、`MIN_EIGEN_RATIO = 3.0`、`NUM_VIEWS = 24`）。
- **holdout 的用途**：只做评测，不参与训练；其 `canonical_geometry.npz` 与自身渲染**配对有效**，可以当 GT。
- **`data_test/` 无配对 3D**：同编号训练网格不能当 GT（见 §8 红线）。

---

## 3. 训练侧

### 3.1 三条线对照表（数值来自 `results/*/config.json` 运行快照）

| | Stage1 SS | Stage2 shape | Stage3 tex |
|---|---|---|---|
| 启动脚本 | `scripts/run_ss_lora_pool1302_canon_axis10.sh` | `scripts/run_shape_lora_pool1302_v2.sh` | `scripts/run_tex_flow_lora_pool1302.sh` |
| 配置 | `configs/gen/ss_flow_lora_64_canon_axis10_pool1302.json` | `configs/gen/shape_flow_lora_1024_v2_pool1302.json` | `configs/gen/tex_flow_lora_1024_pool1302.json` |
| 输出目录 | `results/pool1302_ss_flow_lora_64_canon_axis10` | `results/pool1302_shape_flow_lora_1024_v2` | `results/pool1302_tex_flow_lora_1024` |
| 模型 | `SparseStructureFlowModel` | `ElasticSLatFlowModel` | `ElasticSLatFlowModel` |
| resolution / in_ch | **16** / 8 | 64 / 32 | 64 / **64** |
| model_channels / blocks / heads | 1536 / 30 / 12，rope | 1536 / 30 / 12，rope | 1536 / 30 / 12，rope |
| cond_channels | 1024 | 1024 | 1024 |
| 底模 | `checkpoints/ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors` | `checkpoints/ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16.pt` | `checkpoints/ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.safetensors` |
| 数据集 | `ImageConditionedSparseStructureLatentViews` | `ImageConditionedSLatShape` | `ImageConditionedSLatPbr` |
| 条件图尺寸 | **512** | 1024 | 1024 |
| bs / 梯度累积 | 4 / `batch_split=2` | 2 / 1 | 2 / 1 |
| 训练步数 | 30 000（config 现为 30000；运行快照是 20000，见 §3.5） | 30 000（同上） | 30 000 |
| 存盘间隔 | 2000 | **1000** | 2000 |
| lr | 1e-4 AdamW（wd 0） | 1e-4 AdamW（betas .9/.95） | 同 shape |
| EMA | 0.9999 | 0.9999 | 0.9999 |
| 精度 | amp bf16 | amp bf16 | amp bf16 |
| 显存控制 | — | `LinearMemoryController(target_ratio .75, max_mem_ratio_start .25)` | 同 shape |
| 梯度裁剪 | `AdaptiveGradClipper(max_norm 1.0, p95)` | 同 | 同 |
| `p_uncond` | 0.1 | 0.1 | 0.1 |
| t 采样 | `logitNormal(1.0, 1.0)`, `sigma_min 1e-5` | 同 | 同 |
| snapshot 采样参数 | 12 步 / CFG 7.5 / rescale 0.7 / interval [0.6,1] / rescale_t 5.0 | 12 步 / 7.5 / 0.5 / [0.6,1] / 3.0 | 12 步 / **1.0** / 0.0 / [0.6,0.9] / 3.0 |
| `--data_dir` 关键键 | `base`, `ss_view_latent`, `render_cond` | `base`, `shape_latent`, `render_cond` | `base`, `pbr_latent`, `shape_latent`, `render_cond` |

数据集专有参数：

- SS：`num_views=24`、`crop_scale=1.0`、`latent_frame='canonical'`、`canonical_latent_root='./training_data/pool1302/train/ss_latents_canon'`、`min_view_aspect=0.0`、**`max_view_offaxis=10.0`**、`min_aesthetic_score=4.5`。
- shape：`resolution=1024`、`real_view_prob=0.75`、`max_tokens=32768`、`pretrained_slat_dec=checkpoints/ckpts/shape_dec_next_dc_f16c32_fp16`。
- tex：`full_pbr=true`、`attrs=['base_color','metallic','roughness','alpha']`、`max_tokens=32768`、`pretrained_pbr_slat_dec=checkpoints/ckpts/tex_dec_next_dc_f16c32_fp16`。

### 3.2 LoRA 注入

- 代码：`trellis2/modules/lora.py:17 LoRALinear`、`:64 apply_lora_to_linear_layers`。
- 目标（顺序固定，`scripts/utils/inference_contract.py:14 TARGETS`）：
  `self_attn.to_qkv`、`self_attn.to_out`、`cross_attn.to_q`、`cross_attn.to_kv`、`cross_attn.to_out`。
- 每个 30-block 模型 → **5×30 = 150 层**适配器；rank 8 / alpha 8.0 / dropout 0.0。
- **适配器恒为 fp32**，base 走 bf16 → 训练与推理的数值行为一致；`apply_lora_to_linear_layers` 有幂等保护（重复调用不会重复包裹）。

### 3.3 检查点契约

| 项 | 事实 |
|---|---|
| 存盘 | `trellis2/trainers/basic.py:439 save()` → `ckpts/denoiser_ema0.9999_stepNNNNNNN.pt` + `misc_stepNNNNNNN.pt` |
| 张量数 | 940（150 层 LoRA × 6 张量 + 底座/缓冲） |
| dtype 混存 | SS 30k 有 **220 个 fp32 参数以 bf16 存**（`t_embedder.mlp.*`、`adaLN_modulation.1.*`、`input_layer.*`、`blocks.0.modulation`、`blocks.0.norm2.weight`）；shape 30k 是 640 bf16 + 300 fp32（LoRA 全 fp32） |
| 无损性 | 逐张量按目标模型 dtype 转后与"全 fp32 存盘"版本 **逐位相同**（已实测 SS 220/220、shape 640/640） |
| 续训 | `--ckpt latest` + 30k 配置；每次启动在 `run_attempts/<ts>_<hash>.json` 留完整 command / 配置 / 校验 |
| 启动校验 | `utils/train_launch_contract.py`（显式权重路径、output_dir 保护、自动重试） |

**现状不一致（须知）**：`configs/gen/ss_flow_lora_64_canon_axis10_pool1302_30k.json` 与 `shape_flow_lora_1024_v2_pool1302_30k.json` 在 20k→30k 续训时被用过（见 `run_attempts/*.json` 的 command），但**当前仓库 `configs/gen/` 下已不存在**，只在新克隆归档 `maintenance/completeness_20261019/fresh_clone/configs/gen/` 里有副本；同时两个基础 config 的 `max_steps` 已被改成 30000。⇒ 想复现"从 0 一次跑到 30k"，直接用现在的基础 config 即可；想逐位复现当时的续训序列，要从那份归档里取 30k 配置。

### 3.4 训练成本（实测 per-step，取自各 `log.txt`）

| 线 | 每步 | 30k 合计（含日志/采样开销则更多） |
|---|---|---|
| SS | **0.667 s** @bs4（16³ latent，token 少） | ≈ 5.6 h（20k 段 ≈ 3.7 h，续训 10k 段实测 6793 s ≈ 1.9 h） |
| shape 1024 | ≈ **0.31 s** @bs2（弹性显存） | ≈ 2.6 h（10k 段实测 3732 s） |
| tex 1024 | ≈ **0.45 s** @bs2（`elastic.memory` ≈ 30 GB） | 实测单次 30k = 10 485 s ≈ **2.9 h** |

⇒ "给 SS 再加 10k 步"约 1.9 h；这是判断"值不值得加步数"时的成本基准。

### 3.5 训练侧的两个已知限制

1. **SS 监督覆盖极窄**：`max_view_offaxis=10°` 过滤后，24 视角只剩 **2292/27504 帧（8.33%）**，**平均每对象 2.0 个可用视角**（`results_infer/canonical_orientation_probe_20261019/summary.json`）。加步数等于在同一批 2 张渲染上继续拟合。
2. **SS 目标本身是"每对象一个规范帧 latent"**，而条件图来自同一帧 → 目标与条件强耦合，跨视角泛化没被训练过。

---

## 4. 推理侧

### 4.1 基础管线 `checkpoints/pipeline.json`

| 组件 | 值 |
|---|---|
| `sparse_structure_decoder` | `microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16` |
| `sparse_structure_flow_model` | `ckpts/ss_flow_img_dit_1_3B_64_bf16` |
| `shape_slat_decoder` | `ckpts/shape_dec_next_dc_f16c32_fp16` |
| `shape_slat_flow_model_512` / `_1024` | `ckpts/slat_flow_img2shape_dit_1_3B_512_bf16` / `…_1024_bf16` |
| `tex_slat_decoder` | `ckpts/tex_dec_next_dc_f16c32_fp16` |
| `tex_slat_flow_model_512` / `_1024` | `ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16` / `…_1024_bf16` |
| `sparse_structure_sampler` | `FlowEulerGuidanceIntervalSampler` 12 步 / CFG **7.5** / rescale 0.7 / interval [0.6,1] / rescale_t 5.0 |
| `shape_slat_sampler` | 12 步 / 7.5 / 0.5 / [0.6,1] / 3.0 |
| `tex_slat_sampler` | 12 步 / **1.0** / 0.0 / [0.6,0.9] / 3.0 |
| `image_cond_model` | `DinoV3FeatureExtractor`，`model_name=/home/lyx/lzx/TRELLIS.2/dinov3`（**绝对路径**） |
| `rembg_model` | `BiRefNet`，`/home/lyx/lzx/TRELLIS.2/rmbg` |
| `default_pipeline_type` | **`1024_cascade`** ← 与我们的口径不一致，见 §8 |

### 4.2 `run()` 六步（`trellis2/pipelines/trellis2_image_to_3d.py`）

| # | 步骤 | 代码位置 | 关键事实 |
|---|---|---|---|
| ① | `preprocess_image` | `:166` | RGBA 直接用；否则 BiRefNet 抠图；按 alpha bbox 方裁（`crop_scale=1.0`）；RGB×alpha（背景黑）；最长边 ≤1024 |
| ② | `get_cond` | `:211` | DINOv3 patch tokens + LayerNorm；`neg_cond = zeros_like(cond)`；**会重设 `image_cond_model.image_size`** |
| ②b | `get_ss_cond` | `:238` | 无专用 SS 编码器时等价于 `get_cond(512)` ⇒ **pipeline_type=1024 时 SS 条件仍是 512** |
| ③ | `sample_sparse_structure` | `:270` | 噪声 `(N,8,16,16,16)` → sampler → `decoder(z_s)>0` →（`ss_res` 决定是否 `max_pool3d`）→ `coords = argwhere(decoded)[:, [0,2,3,4]]`，形状 `(N,4)=(batch,x,y,z)` |
| ④ | `sample_shape_slat` | `:324` | `noise = SparseTensor(feats=randn(len(coords),32), coords=coords)` ⇒ **每个 SS 体素 = 一个 shape token**；采样后 `slat*std+mean` |
| ⑤ | `sample_tex_slat` | `:486` | shape_slat 重新归一化作 `concat_cond`；噪声通道 = `tex_in_channels - shape_channels` |
| ⑥ | `decode_latent` / `decode_shape_slat` / `decode_tex_slat` | `:555` / `:460` / `:532` | dual-grid 解码 → `MeshWithVoxel`（`voxel_size=1/res`，`origin=[-0.5]*3`，layout `{base_color:0:3, metallic:3:4, roughness:4:5, alpha:5:6}`）→ `trellis2/representations/mesh/base.py:44 fill_holes()`（cumesh，原地改） |

`pipeline_type` 与 SS 分辨率（`:644` 硬编码字典）：

| pipeline_type | SS coords res | shape 路径 | 最终 res | 是否适配现 LoRA |
|---|---|---|---|---|
| `'512'` | 32 | 512 直采 | 512 | 否 |
| **`'1024'`（当前参考）** | **64** | **1024 直采** | **1024** | **是** |
| `'1024_cascade'` | 32 | 512 → upsample×4 → 1024 | 1024 | 否（token 分布不同） |
| `'1536_cascade'` | 32 | 512 → upsample×4 → 1536 | ≤1536 | 否（另加 96³ 网格外推） |

级联细节：`sample_shape_slat_cascade:366`，up 采样 → `((c+0.5)/lr_res*(hr_res//16)).int()` 量化 → token ≥ `max_num_tokens=49152` 时 `hr_resolution -= 128`（下限 1024）。

### 4.3 LoRA 装载顺序（四个必做动作，缺一即失败或静默错）

```python
from scripts.utils.inference_contract import TARGETS, checked_load, validate_lora
from trellis2.modules.lora import apply_lora_to_linear_layers

m = p.models['shape_slat_flow_model_1024']            # 或 'sparse_structure_flow_model'
n = apply_lora_to_linear_layers(m, TARGETS, rank=8, alpha=8.0, dropout=0.0)   # 1) 注入 150 层
st = torch.load(ckpt, map_location='cpu', weights_only=True)
st.pop('rope_phases', None)                            # 2) 丢掉非参数缓冲
# 3) 逐张量 cast 到目标模型 dtype（关键：SS 有 220 个 fp32 参数以 bf16 存盘）
validate_lora(m, n, st, 8, 8.0, 0.0)                   # 4a) 校验 150 层 + 适配器形状
checked_load(m, st, allow_rope=True)                   # 4b) 严格键校验后加载
```

- `scripts/utils/inference_contract.py:35 checked_load`、`:52 validate_lora`、`:68 glb_coordinates`。
- **不要**跳过 cast：直接 `load_state_dict` 会在 SS 上报 dtype 不匹配；**不要**跳过 `validate_lora`：少挂一层也照样能加载，错误会被静默吸收。

### 4.4 三个可用的推理入口

| 入口 | 用途 | 注意 |
|---|---|---|
| `diagnostics/infer_001_three_arms_v1/infer_001_three_arms.py` | **参考产物生产线**：`SS_DIR`/`SHAPE_DIR`（`:140-141`）指向 30k ckpts，`p.default_pipeline_type='1024'`（`:221`），`assert not ss_swap_yz and not ss_canonicalize`（`:222`），`p.run(..., pipeline_type='1024')`（`:255`） | 不读训练 config，只读 ckpt 路径 + pipeline.json |
| `scripts/infer_finetuned.py` | 通用 CLI（`--sparse_structure_lora` / `--flow_1024_lora` / `--tex_flow_lora` / `--lora_rank` …） | **`--pipeline_type` 默认 `1024_cascade`（`:236`）**，必须显式传 `--pipeline_type 1024` |
| `diagnostics/ss_support_oracle_20261020/ss_support_oracle.py` | 分阶段诊断：只调 `sample_sparse_structure` / `sample_shape_slat` / `decode_shape_slat`，可换 SS 权重（`--ss-ckpt`）、跳过 texture | 用于归因实验（见 §5.3） |

另外 `scripts/utils/evaluate_fixed_frame.py` 是固定帧评测入口；`scripts/utils/inference_contract.py::pipeline_timing` / `local_pipeline_weights` 提供计时与权重哈希清单。

### 4.5 坐标契约（`data_toolkit/canonical_frame.py`，`FRAME_VERSION='trellis.common-frame.gltf-y-up.v1'`）

- raw 轴序 = **(depth, length, up)** = (0, 1, 2)；**长度槽是 y**。
- 导出 GLB：`(x, y, z) → (x, z, -y)`（长度轴落到 z，上方向落到 y）。同类实现见 `inference_contract.py:68 glb_coordinates`。
- 该文件显式声明：**不旋转已生成网格、不转置已编码 latent**；OBJ/未知 up 轴必须另行标定，无隐式回退。
- `run()` 里两个历史补丁开关都必须保持关闭（参考脚本有 assert）：`ss_swap_yz`（盲目换 y/z）、`ss_canonicalize`（逐样本把最长轴置换到轴 1）。当前方案不依赖它们。

---

## 5. 评测侧（口径与当前结论）

### 5.1 统一协议

- 对象：pool1302 **holdout 156**；GT = `data_for_train/<object_id>/canonical_geometry.npz`（与自身渲染配对有效）。
- 条件图：`renders_cond_t/<id>/000_cam_000.png`（侧视）；`steps=12`、`seed=42`、`pipeline_type='1024'`。
- Chamfer：面加权采样 `n=20000`、`seed=4242`、双向、**不做 ICP / 旋转 / 缩放 / 镜像**。
- 比例：extents = bbox 三轴尺寸按最长轴归一（长度轴 ≡ 1），报三轴绝对差之和的均值。
- 朝向：`min(raw, 180°)` 校正项 + `GT self-flip chamfer`（>0.01 才有判别力）；`axis_stats` 对翻转失明，必须成对使用。

### 5.2 已验证结论（可直接引用）

| 实验 | 结论 | 产物 |
|---|---|---|
| shape 20k vs 30k（SS 固定 30k，156 配对） | raw Chamfer 0.0075253 vs 0.0074935，Δ=−3.2e−5，p=0.127；**canonical extents L1 0.0188 vs 0.0184，Δ=−4.0e−4，p=0.023（30k 显著更好）**；翻转 31/156 两臂完全相同 ⇒ 保留 30k | `results_infer/holdout_shape20_vs_30_20261019/` |
| SS 支撑神谕分解（156） | A0（支撑 bbox）0.0431、A2（真实链路）0.0184、B（GT 支撑）0.0066 ⇒ **SS 支撑占 extents 误差 64%**，shape+decoder 地板 0.66%；支撑 bbox 与 GT 完全一致仅 85/156，误差全是整 1 体素且系统性偏胖 | `results_infer/ss_support_oracle_20261020/` |
| SS 20k vs 30k（shape 固定 30k） | extents 0.0196→0.0184（−6.2%，p=0.46 不显著）；支撑 IoU 0.6076→0.6281（p=2.2e−6 显著但仅 +3%）；**bbox 精度 0.0177→0.0169（p=0.54 没动）** | 同上 `aggregate_ss20k_vs_30k.json` |
| 朝向探针 | holdout 翻转率 31/156（19.9%），翻转对象 raw Chamfer 0.0139 vs 未翻转 0.0059 ⇒ 均值约 19% 来自朝向 | `results_infer/canonical_orientation_probe_20261019/` |

### 5.3 归因实验的最小复现集

```bash
# 真实链路 + 神谕 + 支撑 bbox 三臂
python diagnostics/ss_support_oracle_20261020/ss_support_oracle.py \
    --out results_infer/ss_support_oracle_20261020 --gpu 7 --limit 0
# 换 SS 权重做对照
python diagnostics/ss_support_oracle_20261020/ss_support_oracle.py \
    --out results_infer/ss_support_oracle_ss20k_20261020 --gpu 6 --limit 0 \
    --ss-ckpt results/pool1302_ss_flow_lora_64_canon_axis10/ckpts/denoiser_ema0.9999_step0020000.pt
# 汇总（单跑报三臂、双跑报配对比较）
python diagnostics/ss_support_oracle_20261020/aggregate.py \
    --runs results_infer/ss_support_oracle_20261020 results_infer/ss_support_oracle_ss20k_20261020 \
    --labels ss30k ss20k --out results_infer/ss_support_oracle_20261020/aggregate_ss20k_vs_30k.json
```

---

## 6. 系统检查清单（建议每次改动后逐条过）

### A. 数据
- [ ] `diagnostics/pool_targets_v2/COMPLETE.json` 的 `manifest_sha256` 与 `manifest.json` 实算一致、`assets=1302`
- [ ] `SPLIT_PROPOSAL_20261015.json` 的 sha = `d5f5e627…bfd`，train/holdout 不重叠且覆盖全池
- [ ] 每个对象 `renders_cond_t/<id>/` 有 24 张 512² PNG + `transforms.json`，`png_sha256` 可复核
- [ ] `ss_latents_canon/<id>.npz` 键是 `mean`、形状 `(8,16,16,16)`；`ss_view_latents_index/<id>/INDEX.json` 里 `canonical_only=true`
- [ ] `shape_latents/..._1024/<id>.npz` 的 `coords` 与 `pbr_latents/..._1024/<id>.npz` 的 `coords` **逐 token 相等**

### B. 权重
- [ ] ckpt 走的是 `ckpts/denoiser_ema0.9999_stepNNNNNNN.pt`（不是 `misc_*`）
- [ ] 加载后 `validate_lora` 通过：**150 层**、940 张量
- [ ] cast 后无残留 dtype 不匹配（SS 预期 220 个 cast，shape 预期 420 个 fp32→bf16 + 300 个 adapter 不变）
- [ ] 记录 sha256（SS 30k `52853cd5…160b`、shape 30k `bec7e62a…50ed`、shape 20k `b5bff039…5831`）
- [ ] 若换权重：`ss_step` / `shape_step` 与 config 声明一致，且对比时**只改一个变量**

### C. 推理
- [ ] `pipeline_type` 显式传 `'1024'`（不要依赖 `pipeline.json` 的 `1024_cascade` 默认值）
- [ ] `assert not p.ss_swap_yz and not p.ss_canonicalize`
- [ ] SS 条件图是 **512**；shape/tex 条件图是 1024（`pipeline_type=1024` 只把 SS 体素设到 64³）
- [ ] 三阶段 `steps=12`、`seed=42`、CFG 与 `pipeline.json` 一致（7.5 / 7.5 / 1.0）
- [ ] `image_cond_model.model_name` 指向的 `dinov3` 目录存在（绝对路径 `/home/lyx/lzx/TRELLIS.2/dinov3`）
- [ ] 产物是 GLB（`decimation_target=1e6`、`texture_size=4096`、`remesh=True`）；`--geometry_only` / `--fast_glb` 不算正式产物
- [ ] `fill_holes()` 已被调用（`decode_shape_slat` 后）

### D. 评测
- [ ] GT 来自 `data_for_train/<object_id>/canonical_geometry.npz`，**不是** `data_test/` 同编号训练网格
- [ ] Chamfer 协议：面加权、n=20000、seed=4242、无对齐
- [ ] 同时报 `min(raw, 180°)` 与 `GT self-flip`，不要只看 raw
- [ ] 结论落盘到新的结果目录（不覆盖旧目录），并写 manifest（dtype_casts / sha / steps / seed）

---

## 7. 复现命令（当前参考方案）

```bash
cd /data/lzx/TRELLIS.2
PY=/home/test/miniconda3/envs/trellis2/bin/python

# ① 数据（已完成；重跑需数小时，勿随手执行）
$PY diagnostics/pool_package_v1/build_pool1302.py verify

# ② 训练（各约 2–6 h，单卡）
bash scripts/run_ss_lora_pool1302_canon_axis10.sh 6
bash scripts/run_shape_lora_pool1302_v2.sh 7
bash scripts/run_tex_flow_lora_pool1302.sh 6

# ③ 参考产物（001，SS30k + shape30k，1024 直采）
$PY diagnostics/infer_001_three_arms_v1/infer_001_three_arms.py --help

# ④ holdout 评测（156 对象，约 28 min）
$PY diagnostics/eval_holdout_shape20_vs_30_20261019/eval_holdout_shape_20k_30k.py \
    --out results_infer/holdout_shape20_vs_30_20261019 --gpu 6

# ⑤ 归因（见 §5.3）
```

GPU 只用 6/7；GPU 0–3 是他人 vLLM，4/5 常驻 ComfyUI。

---

## 8. 已知坑与红线

1. **`data_test/` 没有配对 3D**。`docs/PROJECT_STATUS.md:10` 明令禁止用同编号训练网格算 Chamfer/宽度误差。因此 `results_infer/001_three_arms_v1/ss30_shape30/arm.json` 里的 `chamfer_vs_gt` 数值无效（其 GT 对不上输入图：`data_test/001.png` 与该对象 24 张渲染的最小平均像素差 64.67）。**只使用 mesh 自洽的度量**（axis/spans/朝向）或改用 holdout。
2. **`pipeline.json` 的 `default_pipeline_type` 是 `1024_cascade`**，而 `scripts/infer_finetuned.py:236` 的 `--pipeline_type` 默认值也是它。cascade 会把 SS 网格降到 32³、并把上采样后的稠密坐标喂给 1024 模型 —— 对现 LoRA 是双重分布外。**任何跑都必须显式 `'1024'`**。
3. **SS 条件恒 512**。`pipeline_type=1024` 只影响 SS 体素分辨率（64³），不影响条件编码器的 `image_size`。把 SS 条件改成 1024 会让 token 数偏离训练分布。
4. **`rope_phases` 必须丢弃**（`checked_load(..., allow_rope=True)`）；否则严格键校验会失败。
5. **`MeshWithVoxel` 没有 `feats` 属性**（`trellis2/representations/mesh/base.py:208`，属性是 `vertices/faces/coords/attrs/voxel_shape/layout`）。`diagnostics/tex_data_v1/accept_tex_lora_30k.py:209-210` 用了旧的 `mesh.feats` ⇒ 三个对象全部 `AttributeError: 'MeshWithVoxel' object has no attribute 'feats'`，`SUMMARY {}`。**tex LoRA 本身是好的**（150 层/940 张量校验通过、三阶段 12 步采样跑完），只是那一步对比读错了属性 —— 纹理验收需要先修这处。
6. **30k 续训配置不在 `configs/gen/`**（见 §3.3）；要用当时的逐位配方，去 `maintenance/completeness_20261019/fresh_clone/configs/gen/` 取。
7. **`pipeline.json` 的 DINOv3/BiRefNet 是绝对路径** `/home/lyx/lzx/TRELLIS.2/{dinov3,rmbg}`，只在本机成立；发布副本用相对 `./dinov3`。
8. **不覆盖已有结果目录**；新实验一律新目录 + manifest。
9. **不碰 `nju/`**；GPU 只用 6/7。
10. **`ss_support64.npz` 不在训练集里**（只存在于 `diagnostics/pool_targets_v2/`），它只是归因实验用的 GT 支撑，别误以为训练集漏了输入。

---

## 9. 待办（与 CURRENT_STATE 对齐，2026-10-26）

- [ ] ① 支撑腐蚀标定实验：把 SS coords 蚀 1 体素后再进 `sample_shape_slat`，验证 extents 能否从 0.0184 降下来（离线已证支撑 bbox 误差可降 19–22%，尚未闭环到 mesh）
- [ ] ② 放宽 `max_view_offaxis` 或改多视角条件（现有 24 视角只用上 ~2 个；深度轴误差占 +0.38 体素是侧视不可观测导致）
- [ ] ③ shape 侧加 ±1 体素支撑抖动训练（支撑错时误差被放大 2.9 倍：0.0098 → 0.0288）
- [ ] ④ 修 `accept_tex_lora_30k.py` 的 `mesh.feats` → `mesh.attrs`，完成带纹理 GLB 验收
- [ ] ⑤ 若要继续加 SS 步数：以 holdout 支撑 bbox 精度为门禁（loss 已与 bbox 精度脱钩）

---

## 10. 文件索引（按阶段）

**数据**
`diagnostics/pool_package_v1/build_pool1302.py`、`diagnostics/pool_package_v1/audit_*.py`、
`data_toolkit/render_cond.py` + `blender_script/render_view.py`（经 `scripts/utils/render_new_data.py`）、
`data_toolkit/canonical_frame.py`、`data_toolkit/build_canon_canonical_latents.py`、`data_toolkit/build_ss_canonical_contract.py`、
`scripts/utils/build_renders_cond_t.py`、`scripts/utils/derive_alpha.py`、`data_toolkit/encode_ss_latent.py`

**训练**
`train.py`（合并 CLI+config、resolve_checkpoint、建模、`load_pretrained_checked`、冻结+`apply_lora_to_linear_layers`、`run_attempts`）、
`trellis2/models/sparse_structure_flow.py:64 SparseStructureFlowModel`、`trellis2/models/structured_latent_flow.py:214 ElasticSLatFlowModel`、
`trellis2/modules/lora.py:17/64`、
`trellis2/datasets/sparse_structure_latent_views.py:19 ImageConditionedSparseStructureLatentViews`、
`trellis2/datasets/structured_latent_shape.py:108 ImageConditionedSLatShape`、
`trellis2/datasets/structured_latent_svpbr.py:313 ImageConditionedSLatPbr`、
`trellis2/trainers/flow_matching/flow_matching.py:354 ImageConditionedFlowMatchingCFGTrainer`、
`trellis2/trainers/flow_matching/mixins/image_conditioned.py:72 DinoV3FeatureExtractor`、`trellis2/trainers/basic.py:439`、
`trellis2/utils/{checkpoint_contract,train_launch_contract}.py`

**推理**
`checkpoints/pipeline.json`、`trellis2/pipelines/base.py:34 from_pretrained`、
`trellis2/pipelines/trellis2_image_to_3d.py`（`:166/:211/:238/:270/:324/:366/:460/:486/:532/:555/:590`）、
`trellis2/pipelines/samplers/flow_euler.py:197 FlowEulerGuidanceIntervalSampler`、
`trellis2/models/sparse_structure_vae.py:112 SparseStructureEncoder / :223 SparseStructureDecoder`、
`trellis2/models/sc_vaes/sparse_unet_vae.py:536 upsample(x, upsample_times)`（级联用）、`trellis2/representations/mesh/base.py:44/208`、
`scripts/utils/inference_contract.py`（`:14 TARGETS`、`:35 checked_load`、`:52 validate_lora`、`:68 glb_coordinates`）、
`scripts/infer_finetuned.py`、`diagnostics/infer_001_three_arms_v1/infer_001_three_arms.py`

**评测**
`diagnostics/eval_holdout_shape20_vs_30_20261019/{eval_holdout_shape_20k_30k.py,analyze.py,make_panels.py,REPORT.md}`、
`diagnostics/ss_support_oracle_20261020/{ss_support_oracle.py,aggregate.py,REPORT.md}`、
`diagnostics/tex_data_v1/{accept_tex_lora_30k.py,sweep_texflow_001.py}`、
`scripts/utils/evaluate_fixed_frame.py`
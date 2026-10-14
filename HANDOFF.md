# TRELLIS.2 火车重建项目 — 交接文档

更新时间：**2026-10-10**

> 本版**就地更新**了 2026-09-12 版（该版未单独存档，其要点已由本版的"已作废/更正"说明与 `REPORT_20261010_lora_dinov3_rebuild.md` 覆盖）。
> 历史上 0907→0912 是另存为 `HANDOFF_20260907.md`，本次未沿用该做法。
> 旧版结论若与本版冲突，**以本版为准**。

本文档面向接手的 agent，交代项目现状、使用方法、注意事项。**继续工作前请完整阅读本文档，
尤其 §6.1 的第 1、4、10 条（都是会让成品"看起来能跑但实际是错的"的坑）。**

---

## 1. 项目概述

基于微软 TRELLIS.2 的本地适配，从单张火车图片（主要是侧视图）重建 3D 模型，
分辨率 1024（`--pipeline_type 1024`，**不是** 1024_cascade）。

- 数据集：designer 编辑过的火车资产，**训练集 1182 个对象**（多视角渲染数据）
- 研究路线：对 TRELLIS.2 的三条 flow 链路分别做 LoRA 微调：
  1. **Stage 1**：稀疏结构 flow（`SparseStructureFlowModel`, res16）
  2. **Stage 2**：形状 shape flow（`ElasticSLatFlowModel`, res64/in32）
  3. **Tex flow**：纹理 PBR 流（`ElasticSLatFlowModel`, res64/in64）
- 推理管线：`scripts/infer_finetuned.py`，可任意组合挂载上述 LoRA

---

## 2. 当前进展（截至 2026-10-10）

### 2.1 各模型的检查点状态（**现存文件为准**）

| 模型 | 目录 | 现存步数范围 | 状态 |
|---|---|---|---|
| SS LoRA (旧, DINOv2 条件训练) | `results/multiview_1186_ss_flow_lora_64_views/ckpts/` | **23 个 EMA，10000–50000** | ⚠️ 见 §6.1-1：训练口径与推理不一致，**不要再用** |
| **SS LoRA (新, DINOv3@512)** | `results/multiview_1186_ss_flow_lora_64_views_dinov3_512/ckpts/` | **50 个 EMA，1000–50000** | ❌ 已训完 50k，但**朝向随输入摇摆，不要上产线**（§6.1-13） |
| SS LoRA (对齐版) | `results/multiview_1186_ss_flow_lora_64_views_alignedyz/ckpts/` | **50 个 EMA** | ❌ 已训完；对齐只是把 29/71 镜像成 67/33，**仍然摇摆**（§6.1-13） |
| SS LoRA (规范帧版, 已终止) | `results/multiview_1186_ss_flow_lora_64_views_canonical/ckpts/` | 到 29100 停 | ❌ 假设已被证伪，见 §6.1-1 |
| Shape Flow LoRA (Stage2) | `results/multiview_1186_flow_1024_lora_v2/ckpts/` | **50 个 EMA，1000–50000** | ⚠️ 原来的 200k 已在 2026-10-09 清理中删除 |
| Tex Flow LoRA | `results/multiview_1186_tex_flow_lora_1024/ckpts/` | 22 个 EMA，1000–130000 | 可用到 130k |

所有检查点均为 `denoiser_ema0.9999_stepXXXXXXX.pt`（7 位补零，EMA，推理用）
+ `denoiser_stepXXXXXXX.pt`（原始权重，避开 EMA 滞后，做诊断时用它）
+ `misc_stepXXXXXXX.pt`（断点续训状态）。

> **2026-10-09 用户下达过清理**：SS 只保留到 50k、tex/ss 之外的历史目录、以及全部
> `multiview_767_*` 目录都已删除。§6.1-8 的"不要清理 results/"**已作废**。

### 2.2 关于"最佳检查点"——旧结论已作废

旧文档写"SS 最佳区间 28000–33000（300k+ 破洞）"。**这个经验现在有了新解释，且不再适用**：

1. **EMA 滞后**：`ema_rate=0.9999`，第 n 步仍保留 `e^{-0.0001n}` 的零初始化
   （2k 时 82%、10k 时 37%、28k 时 6%、50k 时 0.7%）。所以早期检查点"不起作用"
   是 EMA 滞后，不是没学到。
2. **真正的杀手是 y/z 轴未对齐**（§6.1-4）：旧 SS LoRA 学到的坐标系与预训练模型/
   推理管线整整差一次 y/z 交换。配上推理侧的 DINOv3 条件时 LoRA 被压制、朝向看着对
   但 LoRA 收益为零；一旦 LoRA 真正生效，成品会**整体转 90°**。
   所以"28000–33000 最好"本质上是在挑"LoRA 干扰最小"的点位。

**现在的建议（2026-10-11 更正）**：上面"整整差一次 y/z 交换"的说法**已被推翻**（§6.1-13）。
真正的根因是**训练目标本身是混合帧**（71% 一种帧 / 29% 另一种），全局交换在数学上不可能修正。
目前**唯一朝向稳定**的是"旧 33k SS LoRA + 管线默认 DINOv3 条件"（近 base，4/4 正确），
纯 base 次之（002 会退化成 x）。**新的 A@50k / C@50k 不要上产线。**
**不要再参考 28k–33k 这个区间。**

### 2.3 数据与数据管线

| 产物 | 路径 | 说明 |
|---|---|---|
| 训练集 latent | `training_data/multiview_1186/` | 179 G |
| 条件图（原始） | `.../renders_cond/` | 1183 个对象目录；**633 个是不透明白底、549 个是透明真图**（§6.1-10） |
| **条件图（统一透明）** | `.../renders_cond_t/` | 1182 个对象；633 个由 `scripts/utils/build_renders_cond_t.py` 从存档补 alpha，549 个符号链接复用。**新 run 建议改用它** |

  📌 **怎么用**：把 `--data_dir` 里的 `render_cond` 换掉即可，例如
  ```bash
  --data_dir '{"train_data":{"base":"./training_data/multiview_1186","ss_view_latent":"./training_data/multiview_1186/ss_latents_views_mesh_t1_yz","render_cond":"./training_data/multiview_1186/renders_cond_t"}}'
  ```
  验证：`scripts/utils/verify_renders_cond_t.py` —— 新口径下训练侧条件图与推理侧
  相差 |Δ|=1.19/255、相关 0.989；旧口径下不透明对象的灰白背景原样进了网络（非零占比 1.000）。
  ⚠️ 当前正在跑的 `..._alignedyz` run 用的是旧 `renders_cond`，**不要中途替换**。
| SS 逐视角 latent（原） | `.../ss_latents_views_mesh_t1/` | 每对象 12 个 npz；**坐标系未对齐（缺 y/z）** |
| **SS 逐视角 latent（已对齐）** | `.../ss_latents_views_mesh_t1_yz/` | 由 `data_toolkit/encode_ss_view_latents_yz.py` 生成，1182 对象 / 14184 视角 / 1.7 G |
| **SS 逐视角 latent（★ 规范化，推荐）** | `.../ss_latents_views_canon/` | 修正相机约定 + 轴置换，1182 对象 / **28368 视角** / 3.3 G；配套体素 `ss_view_voxels_canon_v4`（§6.1-14） |
| SS 规范帧 latent | `.../ss_latents/ss_enc_conv3d_16l8_fp16_64/` | 每对象 1 个（官方口径），同样未对齐 |
| PBR latent | `.../pbr_latents/tex_enc_next_dc_f16c32_fp16_1024/` | 1182 npz；`full_pbr=true` 过滤后实际 **810** 个样本 |
| 原始资产 | `data_for_train/` | **1323 个单元，114 G**（2026-10-10 新入册 137 个，Concept 401–851） |
| 新入册资产来源 | `new_data/` | 128 G，4 个交付包；`delete/`/`Deleted/` 是否决资产 |

### 2.4 已验证的推理组合与结果

**能用的组合（2026-10-11 更正）**：~~SS(DINOv3@512 版) + `--ss_swap_yz`~~ →
**旧 33k SS LoRA + shape v2 20k**，`--pipeline_type 1024`。

> ⚠️ 下表是 `data_test/001.png` **单图**结果，结论**已被推翻**：A@50k + `--ss_swap_yz`
> 在 002 上仍整体转 90°，003 也预测为 ❌（§6.1-13）。表中数据仅作历史记录保留。

`data_test/001.png` 上的实测（2026-10-10，`shape flow` 固定 20k）：

| SS 检查点 | 不加 `--ss_swap_yz` | **加 `--ss_swap_yz`** | 夹角均值↓ | 夹角p95↓ |
|---|---|---|---|---|
| 10k | z ✅ | y ❌（LoRA 尚弱） | 1.822 | 9.51 |
| 20k | y ❌ | **z ✅** | 1.205 | 4.93 |
| 30k | y ❌ | **z ✅** | 1.069 | 4.83 |
| 40k | y ❌ | **z ✅** | 0.862 | 4.03 |
| **50k** | y ❌ | **z ✅** | **0.855** | **3.52** |
| 旧线上基线（33k + DINOv3 条件） | z ✅ | — | 1.978 | 10.002 |

产物：`results_infer/sweep_ss_dinov3_00X0000_shape20k/`（不加交换）、
`results_infer/yzfix_ssA*_shape20k/`（**加交换，可用版本，50k 最佳**）。

指标脚本：`scripts/utils/mesh_metrics.py`、汇总 `scripts/utils/final_table.py`。

---

## 3. 目录结构

```
/data/lzx/TRELLIS.2/                 # 物理路径；/home/lyx/lzx/TRELLIS.2 是同一份
├── train.py                         # 训练主入口
├── HANDOFF.md                       # 本文档
├── AUDIT_lora_20261009.md           # SS 条件不一致的审计报告
├── REPORT_20261010_lora_dinov3_rebuild.md  # ★ 2026-10-10 重训 + y/z 根因与修复全过程
├── configs/gen/                     # 训练配置（见 §4.2）
├── scripts/
│   ├── infer_finetuned.py           # ★ 推理主入口
│   ├── infer_ss_sweep_generic.sh    # ★ 通用扫描脚本（<ckpt_dir> <step> <gpu> <tag>）
│   ├── infer_ss_dinov3_sweep_gpu.sh # SS 扫描（DINOv3 版）
│   ├── run_ss_lora_dinov3_512_50k.sh        # ★ SS 重训（DINOv3@512）
│   ├── run_ss_lora_alignedyz_dinov3_50k.sh  # ★ SS 重训（y/z 对齐版，治本）
│   ├── run_stage2_lora_v2*.sh / run_tex_flow_lora.sh   # shape / tex 续训
│   ├── prepare_pbr_pipeline.py      # PBR 数据管线
│   ├── utils/                       # ★ 工具（下表）
│   ├── inference_history/ training_history/            # 历史脚本
│   └── run_*.sh
├── trellis2/                        # 核心代码（datasets/ trainers/ modules/ pipelines/）
├── data_toolkit/                    # 数据准备工具（本地改动较多）
├── o-voxel/                         # 体素化/网格导出（本地编译）
├── checkpoints/ckpts/               # 官方预训练权重（22 G，勿动）
├── results            -> /data/TRELLIS.2/results          （1.4 T）
├── training_data      -> /data/TRELLIS.2/training_data    （179 G）
├── data_for_train/                  # 原始资产（114 G）
├── new_data/                        # 新到货交付包（128 G）
├── results_infer/                   # 推理产物
└── nju/                             # TRELLIS.1 资产（561 G，勿动，见 §7）
```

`scripts/utils/` 里本项目新增的关键工具：

| 脚本 | 用途 |
|---|---|
| `stage_new_data.py` / `validate_glb.py` / `promote_new_data.py` | 交付包 GLB 入册流程（暂存→校验→移入 data_for_train） |
| `render_new_data.py` | 批量渲染 24 视角（Blender 4.5.1 + Cycles GPU，支持 `--transparent`） |
| `build_renders_cond_t.py` | 由存档不透明渲染推导 alpha，生成统一透明的条件图集 |
| `mesh_metrics.py` / `final_table.py` / `sweep_report.py` | 成品几何指标（二面角）与汇总表 |
| `ascii_silhouette.py` | 把渲染图变成 ASCII 剪影，用于无法传图时核对形状与朝向 |
| `ss_yz_swap_test.py` | 判别 SS 目标是否相对管线约定做了 y/z 交换 |
| `ss_frame_probe.py` | 判定模型输出落在"视角帧"还是"规范帧" |
| `probe_watch.sh` / `sweep_watch*.sh` | 等检查点出现自动跑探针/扫描 |

---

## 4. 训练方法与配置

### 4.1 通用命令模板

```bash
cd /data/lzx/TRELLIS.2
/home/test/miniconda3/envs/trellis2/bin/python train.py \
  --config configs/gen/<config>.json \
  --output_dir results/<exp_name> \
  --ckpt latest \        # 或 none（从头，LoRA 从预训练权重起）/ 具体步数
  --num_gpus 1 \
  --data_dir '{"train_data":{"base":"./training_data/multiview_1186", ...}}' \
  --auto_retry 3
```

- `--data_dir` 是 JSON 字符串，键名必须与 config 里 dataset 类需要的 key 匹配
- `--ckpt latest` 从 `misc_step*.pt` 最大步数恢复（含 optimizer/EMA/sampler 状态）
- 日志 `<output_dir>/log.txt`（append，格式 `步数: {json}`）；**注意这个文件会被多次
  resume 的进程追加，"末条 elapsed/step" 不等于真实速度**——判断速度要看单段的
  `time.step` 中位数
- 检查点 `<output_dir>/ckpts/`，文件名 `denoiser_ema0.9999_step{步数:07d}.pt`
- `train.py` 会把 `command.txt` 与 `config.json` 写进 output_dir（**复现以这两份为准**）

### 4.2 三条链路的配置

| 模型 | config | data_dir 额外 key | 备注 |
|---|---|---|---|
| SS（旧, 勿用） | `ss_flow_lora_64_views_train_to_1000k.json` | `ss_view_latent: .../ss_latents_views_mesh_t1` | 条件用 DINOv2Local@518，与推理不一致 |
| **SS（DINOv3@512）** | `ss_flow_lora_64_views_dinov3_512_50k.json` | 同上 | 条件改 DINOv3@512；`image_size 512`；`crop_scale 1.0`；max_steps 50k |
| **SS（y/z 对齐版）** | `ss_flow_lora_64_views_alignedyz_dinov3_50k.json` | `ss_view_latent: .../ss_latents_views_mesh_t1_yz` | 治本方案 |
| Shape Stage2 | `shape_flow_lora_1024_v2_train.json` | `shape_latent: .../shape_enc_next_dc_f16c32_fp16_1024` | max_steps 200000（现存到 50k） |
| Tex Flow | `tex_flow_lora_1024_train.json` | `pbr_latent: ...` **和** `shape_latent: ...` | `ImageConditionedSLatPbr`、`full_pbr=true` |

三条链路公用：LoRA rank 8 / alpha 8 / 5 个 attention 目标、lr 1e-4（AdamW wd0）、
EMA 0.9999、`p_uncond` 0.1、`logitNormal(1,1)`、bf16 amp、`AdaptiveGradClipper(1.0,95)`。
batch 见各自 config（SS 为 4/split2，shape/tex 为 2）。

2026-10-10 新增的两个可调项：

- **`snapshot_sampler_params`**（trainer 参数）：把 `i_sample` 快照的采样参数对齐线上
  `checkpoints/pipeline.json`，否则**训练时看到的图不代表线上效果**。
  三份配置都已写入（SS: 12步/g7.5/rescale0.7/[0.6,1.0]/rescale_t5.0；shape:
  12步/g7.5/rescale0.5；tex: 12步/g1.0/rescale0/[0.6,0.9]）。
- **数据集的 `crop_scale` / `latent_frame` / `skip_snapshot`**（`ImageConditionedSparseStructureLatentViews`）。
  `crop_scale=1.0` 与官方 `hsize=max/2` 及推理 `preprocess_image` 一致；旧 run 是 1.2。

### 4.3 常用 GPU

GPU 0–3 常被 `VLLM::Worker_TP0` 占满；**4–7 一般可用**，但 GPU7 偶尔被
`vbr.sam31_keyframes` 占用，启动前先看 `nvidia-smi`。训练用 `CUDA_VISIBLE_DEVICES` 指定。

---

## 5. 推理方法

### 5.1 主入口

```bash
CUDA_VISIBLE_DEVICES=5 /home/test/miniconda3/envs/trellis2/bin/python -u scripts/infer_finetuned.py \
  --input_dir data_test --output results_infer/<exp_name> \
  --pipeline_type 1024 \
  --vae_decoder checkpoints/ckpts/shape_dec_next_dc_f16c32_fp16.pt --vae_only \
  --sparse_structure_lora results/multiview_1186_ss_flow_lora_64_views_dinov3_512/ckpts/denoiser_ema0.9999_step0050000.pt \
  --ss_lora_rank 8 --ss_lora_alpha 8 --ss_swap_yz \
  --flow_1024_lora results/multiview_1186_flow_1024_lora_v2/ckpts/denoiser_ema0.9999_step0020000.pt \
  --shape_lora_rank 8 --shape_lora_alpha 8 \
  --no_video
```

- `--ss_swap_yz`：**用"未对齐"数据训出来的 SS LoRA 时必须加**（见 §6.1-4），
  且要与足够训练过的检查点搭配（本配置约 ≥20k）。
  用 `ss_latents_views_mesh_t1_yz` 训出来的模型**不要加**。
- `--ss_cond_model dinov2`：让 SS 阶段单独用 DINOv2Local@518 + 1.2 裁剪（复刻旧 run 口径）。
  仅用于诊断，**不建议**作为产线配置（配旧 LoRA 会得到转 90° 的成品）。
- 三个 LoRA 可任意组合/省略；`--no_video` 建议保留（视频渲染慢）
- 输出目录有 `run.log`（运行记录）与 `infer.log`（详细日志）

### 5.2 LoRA 加载细节（踩过的坑）

- 加载 SS/tex LoRA 前会 `state_dict.pop('rope_phases')`——它是运行时确定性 buffer，
  检查点里的 bf16 副本会破坏它导致空坐标（已修）。加载时 `missing=1` 是正常的。
- LoRA 注入用 `apply_lora_to_linear_layers`，rank/alpha 必须与训练一致（当前 8/8）
- 训练侧 `_init_image_cond_model` 会把条件编码器转 **bf16**，推理侧保持 **fp32**。
  实测对本项目结论无影响（§6.1-4 的 A/B 两种精度结果一致），但知道有这处差异。

---

## 6. 注意事项（重要！）

### 6.1 已知问题与坑

1. **SS 的"坐标系"问题（最重要，务必先读）**
   本地逐视角 SS 训练目标与预训练模型/推理管线约定**整整差一次 y/z 交换**。
   实测：`IoU(预训练预测, 本地GT) = 0.026`，`IoU(swap_yz(预训练预测), 本地GT) = 0.217`
   （8~9 倍）。后果：SS LoRA 一旦真正生效，成品就**整体转 90°**（bbox 长轴从 z 变 y，
   渲染出来是一块竖直的板）；而在它被压制时（例如旧 LoRA 配 DINOv3），朝向看着对、
   但 LoRA 的收益为零——**这就是"细节不足、边缘不流畅"的根**。
   两种解法：
   - 推理侧 `--ss_swap_yz`（已实现并验证，配未对齐数据训出的 LoRA 用）
   - 数据侧用 `ss_latents_views_mesh_t1_yz`（已生成）重训（治本，推理不需要开关）
   详见 `REPORT_20261010_lora_dinov3_rebuild.md` §5.8/§5.9。
   ⚠️ 附带更正：2026-10-10 曾一度把该问题误判为"视角帧 vs 规范帧"监督口径差异，
   并据此跑了 `..._canonical` run；后经实测证伪（两者只差一次绕长度轴的滚转，
   该 run 在 20k 同样竖直），已在报告 §5.0 更正、run 于 29.1k 终止。**不要再走这条路。**
2. **破洞问题**：旧 SS LoRA 步数过大（300k+）会稀疏化过度 → mesh 破洞。
   但现在已知"最佳 28k–33k"主要是 EMA 滞后 + 上述 y/z 问题的产物，**不要再用那个区间**。
3. **tex flow 10 万步以内涂装改善有限**：6 组消融在 `results_infer/data_test_ss33k_shape20k_texflow*/`。
   续训用 `scripts/run_tex_flow_lora.sh`（已配 `--ckpt latest`，config `max_steps=300000`）。
4. **坐标对齐（PBR/dual_grid）**：o_voxel 的 `blender_dump_to_volumetric_attr` 与 dual_grid
   的体素坐标轴需要 y/z 交换，`prepare_pbr_pipeline.py` 的 `fix_dump`/`align_to_dual_grid`
   已处理，训练数据集断言 shape/pbr coords 严格相等。
   📌 **但 SS 的逐视角体素化脚本（`data_toolkit/generate_ss_view_voxels_mesh_t1.py`
   及 v2 系列）没有做这个对齐** —— 这正是第 1 条的来历。改这几个脚本时务必同步处理。
5. **白膜数据**：少量无贴图资产已由 `full_pbr=true`（要求三张贴图齐全）过滤，1182 → 810。
6. **训练首步 snapshot 崩溃**（已修）：`SLatPbr.visualize_sample` 用
   `cv2.imread('assets/hdri/forest.exr')`，pip 版 OpenCV 无 EXR 解码器 → 崩溃。
   已在数据集加 `skip_snapshot` 并在 trainer 主循环检查。
   📌 **现状**：SS 与 tex 数据集 `skip_snapshot=True`（**根本不出快照**），
   只有 shape 会出。原因：主循环里 `self.snapshot()` 未包 try，失败会终止训练。
   已实测 SS 传 `skip_snapshot=false` 可正常产出（含线上口径参数），需要时再开。
7. **GPU7 可能被外部占用**：`vbr.sam31_keyframes` 时不时占用 GPU7。
8. ~~**不要清理 results/ 历史目录**~~ **已作废**：用户在 2026-10-09 明确要求清理
   （SS 只留到 50k、删除 `multiview_767_*` 等）。`nju/` 仍然**不要动**。
9. **不要 git reset/checkout**：大量本地修改未提交，重置会丢工作。
   `configs/gen/*`、`trellis2/`、`data_toolkit/` 多为未跟踪或有本地改动。
10. **条件图有两套约定，别混**
    `renders_cond/` 里 **633 个是不透明白底（alpha 恒 255）**的 Blender 渲染，
    **549 个是透明（2048² 真图）**。不透明那批会让训练侧的 alpha 裁剪退化
    （crop 越界填充、物体缩到约 71%），与推理侧"100% 裁剪 + 黑底"不一致。
    已生成统一透明版 `renders_cond_t/`（633 个补 alpha + 549 个符号链接），**新 run 建议用它**。
    ⚠️ 不要试图"重渲"那 633 个：`data_for_train` 里 **327 个 .obj 的 .mtl 全部缺失（0/327）**，
    重渲会得到无贴图灰模，比存档更差。
11. **渲染约定：色彩管理必须钉死 Filmic**
    存档渲染是 Blender 3.x（默认 Filmic）产物；Blender 4.x 默认 AgX，会让色调整体偏移
    约 4%（白底 0.806 → 0.757，逐像素 |Δ| 9.88/255 vs Filmic 的 0.74/255）。
    `data_toolkit/blender_script/render_view.py` 已钉死 `sRGB/Filmic/look=None`；
    官方 `render_cond.py` 用 `film_transparent=True`，本地新版 `render_view.py` 也已支持。
12. **Blender 4.x 移除了 `import_scene.obj`**：`load_object` 已改为优先 `wm.obj_import`。

### 6.1-13 ⚠️ 朝向问题的真正根因：训练目标是混合帧（2026-10-11）

**先记住结论**：朝向问题**不是**"缺一个全局 y/z 交换"，**任何全局交换都不可能修好**。
根因是**逐视角 SS 训练目标里同时存在两种坐标系**。

**1. 旧结论失效的原因是样本量。** 所有 sweep 目录（`sweep_ss_*`、`yzfix_ssA*`）里
**只有 `001.glb`**，"某配置朝向稳定"全都建立在 1 张图上。

**2. 补齐 002/003/004 后**（成品 GLB 最长轴，应为 z）：

| 输入 | 纯 base SS | **旧 33k + DINOv3 条件** | A@50k + `--ss_swap_yz` |
|---|---|---|---|
| 001 | z ✅ | z ✅ | z ✅ |
| 002 | **x ❌** | z ✅ | **y ❌** |
| 003 | z ✅ | z ✅ | remesh 卡死 |
| 004 | z ✅ | z ✅ | z ✅ |

只有"旧 33k + DINOv3 条件"4/4 正确。

**3. 训练目标是混合帧**（实测，见 `results_infer/_ss_orient_datatest.log`）：

| 数据 | n | 长度轴 = h（正确） | 长度轴 = w（错） |
|---|---|---|---|
| 原始 per-view 体素 | 1182 | 29% | **71%** |
| 原始视角帧 latent | 300 | 33% | **67%** |
| yz 对齐 latent | 300 | **67%** | 33% |

对齐目录确实等于原始目录的精确转置（`IoU(对齐, 原始转置)=1.000` / 未转置 `0.040`），
但**全局交换只是把 29/71 镜像成 67/33，混合帧没有被消除**。

**4. 下游期望的 SS 长度轴是 h，不是 z。** `sweep_report.py` 文档写的"长度轴 = z"是错的。
`--ss_swap_yz` 做的正是 h↔w 交换，所以它只是在给"已训歪"的模型打补丁。

**5. 为什么旧 33k 稳定**：EMA 残差 3.7%，基本等同 base，继承预训练的稳定帧——
**是巧合，不是正确性**。A/C 训到 50k（残差 0.7%）真正主导输出后，混合帧的毒性才显现。

**6. 修法**：做**逐对象朝向规范化**（把每个资产长度轴统一到 h），再重新逐视角体素化 →
重新编码 → 重训。在那之前，用旧 33k 或纯 base 出成品。

**7. 顺带确认**：预训练 SS 模型**期望 DINOv3 条件**——换成 DINOv2@518 后连 base 都塌成
0 体素。旧 run 用 DINOv2 训练属真实缺陷；但 2026-09-11 的旧基线在推理时用的其实是
管线默认 DINOv3（"训练 DINOv2 / 推理 DINOv3"错配），错配把模型压回 base 稳定帧。

### 6.1-14 ★ 朝向问题已定位并修复：体素化绕错了轴（2026-10-11）

**一句话**：逐视角体素化脚本绕 dump 帧的 **Z** 轴旋转，而渲染相机的环绕轴不是它；
占 **71%** 的资产长度轴恰为 Z，于是这些对象的 latent **绕自身长度轴自转**——12 个视角
几乎完全相同，而渲染图是正常环绕（侧视 6.91 → 端视 0.90）。**图文彻底不匹配**，
模型因此学不到稳定坐标系。详见 REPORT §11。

**产物（有效版本）**
| 项 | 路径 |
|---|---|
| 修正版体素化脚本 | `data_toolkit/generate_ss_view_voxels_canon.py` |
| 编码脚本（含轴置换对齐管线约定） | `data_toolkit/encode_ss_view_latents_canon.py` |
| 逐视角体素 1182×24 | `training_data/multiview_1186/ss_view_voxels_canon_v4/` |
| 逐视角 latent 1182×24 | `training_data/multiview_1186/ss_latents_views_canon/` |
| 训练脚本 | `scripts/run_ss_lora_canonv2_dinov3_50k.sh` |
| 输出 | `results/multiview_1186_ss_flow_lora_64_views_canonv2/` |
| 数据验证 | `scripts/utils/verify_canon_data.py` |
| 训练后评测 | `scripts/utils/ss_eval_canon.py` |

**关键参数**（由 `scripts/utils/canon_transform_search.py` 标定）
```
顶点轴映射 (0,2,1)、符号 (-1,+1,-1)、环绕角偏移 90
编码时再施加循环置换 (2,0,1)
```
> ⚠️ 只做 y/z 交换会引入**镜像**（对换行列式 −1）；标定结果是纯旋转。

**验证**（30 对象 × 8 视角）：

| 指标 | 旧数据 | 规范化后 |
|---|---|---|
| 渲染宽高比 vs latent(E1/E0) 相关 | **−0.492** | **+0.971** |
| latent 长轴对视角不变的对象 | **73%** | **0%** |

**可删的废弃中间产物**：`ss_view_voxels_canon`(镜像)、`_v2`(双重置换 bug)、
`_v3`(offset 变量名冲突 bug)。**只保留 `_v4`。**

**两个并行对照 run（2026-10-11 启动，各 50k/约 10h）**

| run | 目标口径 | 数据 | 输出 |
|---|---|---|---|
| A′ 逐视角（相机帧） | 长度轴随视角转 | `ss_latents_views_canon` | `results/multiview_1186_ss_flow_lora_64_views_canonv2/` |
| B′ 规范帧 | 长度轴恒 index1 | `ss_latents_canon` | `results/multiview_1186_ss_flow_lora_64_views_canon_canonical/` |

实测原厂 base 在**侧视**输入上 93% 输出 index1（与修正后 GT 100% 一致），
但在**端视**上只有 60%（base 本身欠定）。

> **验收判据（用户 2026-10-11 指示）**：沿用生产默认**固定种子 `--seed 42`**即可，
> 不要求所有种子；以最终模型效果为准。多种子结果作鲁棒性参考。
> 兜底可用已验证的 `--ss_canonicalize`（逐样本把最长轴摆到下游期望轴，对任何分支成立）。

评测判据已修正为 `ratio = E1 / max(E0, E2)`（>1.5 明确正确 / <0.67 明确错误，含俯仰错 /
其余斜视），只用 argmax 或 E1/E2 都会误判。详见 REPORT §11.16–11.18 与 §12。

### 6.2 环境信息

- Python：`/home/test/miniconda3/envs/trellis2/bin/python`（所有训练/推理都用它）
- **Blender：`/data/lzx/opt/blender-4.5.1-linux-x64/blender`**
  （旧文档写的 `/tmp/blender-4.5.1-linux-x64` **已被清空**；`/tmp` 会被清，别再往那装）
- 磁盘：`/data` 7 T 已用 **5.5 T（84%）**；`/` 已用 559 G（67%）
- 项目路径两个挂载点：`/data/lzx/TRELLIS.2`（物理）与 `/home/lyx/lzx/TRELLIS.2`（逻辑），**用 /data/lzx**

### 6.3 当前运行中的任务

**2026-10-11 16:30 时点（最新）**：

**唯一在跑的训练是 B′** `results/multiview_1186_ss_flow_lora_64_views_canon_canonical`
（主进程 PID 4098953，GPU5）。此刻 step 25000 / 50000，1.47 steps/s，ETA 约 21:06 UTC。

**A′** `..._canonv2` **已于 16:2x 按用户指示终止**（进程组 3991800，停在 step 28000，
`ckpts/` 139 G 保留未删）。原因：它在 **step 14000 从正确分支翻到错误分支并卡住约 14000 步**，
4 图 × 8 种子评测 **0/32 = 0%**；而 B′@20k 同口径 **32/32 = 100%**。详见 REPORT §11.19。

watcher（PID 4157789，`/tmp/eval_watch.sh`，`CUDA_VISIBLE_DEVICES=7`）仍在运行，
会对 B′ 的 40k / 50k 自动续评。已完成：`A'20000`、`B'20000`。

GPU4、GPU6 空闲；GPU0–3 被他人占用（各 134 GB，0% 利用率）；`/data` 剩 677 G。

**⚠️ 最重要的一条**：候选检查点是 **B′（规范帧目标）**，不是 A′ —— 逐视角目标已证伪。
001 上迄今最好成品是 **B′@24k**（`results_infer/newrun_B24k_001/001.glb`：
长度轴 z ✔、轮廓 IoU 0.766、二面角 1.117 / p95 4.16）。
归档见 `results_infer/latest_20261011/`（先读其中的 `MANIFEST.md`）。

**验收时不得加 `--ss_canonicalize`**，否则"成品长度轴 = z"这条判据恒真、等于没测。
**⚠️ 2026-10-11 16:40 更正**：B′@24k 的 001 成品长度轴虽是 z，但长高比 **8.91 vs 输入图 4.22**（2.11×），比旧基线 8.57 更瘦，沿车长的截面几乎是平的。轮廓 IoU 0.766 是斜视挑选造成的假象，不能当验收。规范帧目标把长度钉在 64 格（训练集长/截面中位 5.33），模型越训越瘦。详见 REPORT §11.21。


<details><summary>2026-10-10 收尾时点（存档，已过期）</summary>

### 6.3-存档 当前运行中的任务（2026-10-10 收尾时点）

- **SS 对齐版重训**：`scripts/run_ss_lora_alignedyz_dinov3_50k.sh 5`（GPU5），
  输出 `results/multiview_1186_ss_flow_lora_64_views_alignedyz/`
- 已挂的 watcher（会自动产出中间结论，日志在 `results_infer/_*.log`）：
  `aligned_sweep_watch`（对齐版 10k–50k 扫描）、`probe_watch.sh A`、
  `/tmp/aligned_probe.sh`（帧归属探针）
- GPU 4/6/7 空闲

接手后建议方向：
- **待办 1（最高）**：等对齐版训到可用步数（EMA 需 ≥20k，建议看 30k/50k），
  用 `scripts/infer_ss_sweep_generic.sh <对齐版ckpt目录> <step> <gpu> ss_aligned`
  验证"不需要 `--ss_swap_yz` 也朝向正确"，再与 §2.4 的 `--ss_swap_yz` 路线对比。
- **待办 2**：把新入册的 137 个资产（`data_for_train` 里的 `Concept*_*_P`）
  跑通 latent 编码，纳入训练集（当前训练集仍是 1182 个）。
- **待办 3**：`Concept834_..._P.glb` 是 132 字节空壳导出，可从对应 `.blend` 重导后补入库。
- **待办 4**：观察到一个疑点——SS 逐视角目标的长轴分布不均匀
  （对齐前 `{z:5,y:3}`、对齐后 `{y:4,z:2}`），说明**源资产的朝向可能不一致**，
  会让"某视角→规范帧"的映射不稳定。值得单独查证。

---

</details>

## 7. nju/ 目录说明（不要动）

`nju/` 是 TRELLIS.1 相关资产：`3d_gen_online`（活跃的 FastAPI 服务代码，合并 LoRA 权重在
`nju/3dgen_train/checkpoints/TRELLIS-image-large-lora`）、`trellis-lora`（TRELLIS.1 LoRA
参考实现）、`3dgen_train`（历史训练产出）、`hy3d`（数据筛选，与 TRELLIS 无关）。
约 **561 G**，用户明确保留。

---

## 8. 快速验证清单（接手自检）

```bash
cd /data/lzx/TRELLIS.2
# 1. 环境
/home/test/miniconda3/envs/trellis2/bin/python -c "import torch; print(torch.cuda.device_count())"
# 2. 检查点完整性
ls results/multiview_1186_ss_flow_lora_64_views_dinov3_512/ckpts/ | grep ema | tail -2   # 到 50k
ls results/multiview_1186_flow_1024_lora_v2/ckpts/          | grep ema | tail -2          # 到 50k
ls results/multiview_1186_tex_flow_lora_1024/ckpts/         | grep ema | tail -2          # 到 130k
ls results/multiview_1186_ss_flow_lora_64_views_alignedyz/ckpts/ | grep ema | tail -2     # 对齐版进度
# 3. 数据集
ls training_data/multiview_1186/renders_cond_t | wc -l                       # ~1182
ls training_data/multiview_1186/ss_latents_views_mesh_t1_yz | wc -l          # 1182
ls training_data/multiview_1186/pbr_latents/tex_enc_next_dc_f16c32_fp16_1024 | wc -l  # ~1183
# 4. GPU
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
# 5. 冒烟推理（1 张图，约 6 分钟）——注意 --ss_swap_yz 与检查点的搭配
CUDA_VISIBLE_DEVICES=5 /home/test/miniconda3/envs/trellis2/bin/python -u scripts/infer_finetuned.py \
  --image data_test/001.png --output /tmp/smoke_handoff --pipeline_type 1024 \
  --vae_decoder checkpoints/ckpts/shape_dec_next_dc_f16c32_fp16.pt --vae_only \
  --sparse_structure_lora results/multiview_1186_ss_flow_lora_64_views/ckpts/denoiser_ema0.9999_step0033000.pt \
  --ss_lora_rank 8 --ss_lora_alpha 8 --no_video   # 旧 33k；不要加 --ss_swap_yz（§6.1-13）
# 6. 核对朝向（成品长度轴应为 z）+ 几何指标
/home/test/miniconda3/envs/trellis2/bin/python scripts/utils/mesh_metrics.py /tmp/smoke_handoff/001.glb
/home/test/miniconda3/envs/trellis2/bin/python scripts/utils/final_table.py
```
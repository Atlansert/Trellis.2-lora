# TRELLIS.2 SS LoRA 交接（2026-10-12）

给下一个没有参加过此前对话的 AI。先读文末「新会话启动摘要」，再按节工作。
旧文档 `HANDOFF.md`（2026-10-11 18:11）和 `REPORT_20261010_lora_dinov3_rebuild.md` 的 §11.20、§11.22 **不是当前状态**。以本文件和磁盘为准。

可信度标记：`[代码]` 本会话读过源码或配置；`[日志]` 训练/推理日志；`[实测]` 本会话对现存文件重新算过；`[报告]` 只来自 REPORT，本会话未重跑；`[讨论]` 对话中的判断，未用新实验证实。

---

## 0. 新会话启动摘要

项目在 SSH 服务器 `/data/lzx/TRELLIS.2`（用 `/data/lzx`，不要用 `/home/lyx/lzx`）。Python：`/home/test/miniconda3/envs/trellis2/bin/python`。调用 SSH 工具时不要传 `connection_id`。

要做的事：让本地列车资产的稀疏结构 LoRA，在 `data_test/001–004`、生产默认 `--seed 42`、**不加** `--ss_canonicalize` 时，成品 GLB 长度轴为 z，且几何不比旧基线差。多种子只作参考。

实际跑完的是**逐视角相机帧**侧视 LoRA，不是 REPORT §11.22 写的规范帧 run：

- 配置 `[代码]` `configs/gen/ss_flow_lora_64_views_side_dinov3_20k.json`
- 脚本 `[代码]` `scripts/run_ss_lora_view_side_dinov3_20k.sh`
- 输出 `[日志]` `results/multiview_1186_ss_flow_lora_64_views_side`，20000/20000，2026-10-12 08:25 UTC，3.73 h，`log.txt` 末步 elapsed 13412 s，loss mse 0.0307（首步 0.0437）
- 数据：1151 对象，15502 视角对，`latent_frame=view`，`min_view_aspect=3.5`，`max_view_offaxis` 未设置
- 推理固定形状 LoRA `results/multiview_1186_flow_1024_lora_v2/ckpts/denoiser_ema0.9999_step0020000.pt`
- 001 / seed 42 的 AABB `[实测]`：2k 与 10k 长轴仍是 z（z/y ≈ 6.44–6.47）；12k 仍是 z 但宽度已塌（x/y 1.244 → 0.937）；14k 变成 yz 斜向（y 0.585，z 0.890，z/y 1.52，argmax 仍是 z）；16k 起 argmax 为 y。用户目视 20k 车身严重扭曲，不只是翻轴。
- 没有任何训练进程在跑。GPU 0–3 各约 134 GB、利用率 0（别人的）；4–7 空闲。不要占用 0–3。
- `nju/` 禁止修改。禁止 `git reset` / `git checkout`。不要为了验收打开 `--ss_canonicalize`。

当前没有已验证的修复。最值得先做的是查清那 15502 对里有多少是「图很像侧视、目标朝向却差几十度」，再决定要不要用数据集里已经写好、但这次没开的 `max_view_offaxis`。这仍是假设，不是结论。

---

## 1. 项目目标

最终问题：TRELLIS.2 原厂稀疏结构（SS）底座在这批列车侧视照片上朝向大体对，但几何不够像目标资产。希望用 LoRA 把 SS 拉向本地数据，同时不破坏「成品长度轴 = z」和车身比例。形状 flow、纹理不是本阶段主目标。

输入：一张 RGB 条件图（验收用 `data_test/001.png`–`004.png`，518²，无 alpha）。`[实测]` 以 `max(RGB)>16` 取前景：001 宽高比 4.23，002 3.22，003 3.93，004 5.69。REPORT §11.21 写的 001 = 4.22 是另一种前景定义，差 0.01，可以当作同一事实。

输出：`scripts/infer_finetuned.py` 导出的 GLB。SS 采样 12 步，形状 SLat 12 步，然后 remesh。

核心评价（用户 2026-10-11，REPORT §11.16，同日 §11.21 收紧）`[报告]`：

- 生产默认 `--seed 42`。
- `data_test/001–004` 的 GLB 长度轴都是 z。
- 几何不劣于旧基线；至少要看侧视长高比相对输入图，不能只看轴，也不能用轮廓 IoU 或二面角当验收。
- 多种子只记录，不作门槛。
- 验收时禁止 `--ss_canonicalize`。它按最长轴事后旋转，会让「长度轴 = z」恒真。

当前阶段：view-side 这轮 20k 已经训完，001 上失败。002–004 **没有**这轮的推理产物。不要把 001 的失败自动写成四张图都测过。

---

## 2. 当前技术架构

### 2.1 流程

```
条件图
  → DINOv3@512（crop_scale 1.0）
  → SS flow（底座 + LoRA）采样 16³ × 8 通道 latent
  → SS decoder 得到占用
  → 冻结的 shape flow LoRA（1024，step 20000）
  → shape VAE decoder
  → remesh → GLB
```

纹理 flow 在上述推理日志里仍然跑了 12 步，但本阶段不评估贴图。`[日志]` `results_infer/viewside_*_001/infer.log`。

已核实的轴约定 `[代码]` `data_toolkit/encode_ss_view_latents_canon.py`：

- 体素相机帧：index0 = 画面水平，index1 = 视深，index2 = 画面竖直。
- 编码前对坐标做循环置换 `(2,0,1)`，使纯侧视的长度落到 index1。
- 脚本注释写明：底座/管线约定是 (竖直, 长度, 视深)，原厂 base 侧视输出类似 `(10, 64, 10)`。
- `[报告]` §11.19：原始 SS 的 dim1 最长 <=> GLB 长度轴 z。本会话没有重解码 SS occupancy，只量了 GLB 的 AABB。

### 2.2 模型与依赖

| 角色 | 路径 / 身份 | 依据 |
|---|---|---|
| SS 底座 | `checkpoints/ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors`。1536 通道，30 层，12 头，RoPE，bf16 | `[代码]` 配置；`[日志]` 已加载 |
| SS LoRA | rank 8，alpha 8，dropout 0。目标 `self_attn.to_qkv/to_out`、`cross_attn.to_q/to_kv/to_out`。150 层，可训练 4,669,440 / 1,296,849,416 | `[日志]` |
| 条件 | `DinoV3FeatureExtractor`，`./dinov3`，512 | `[代码]` |
| 形状 LoRA | `results/multiview_1186_flow_1024_lora_v2/.../denoiser_ema0.9999_step0020000.pt`（2026-09-09，2.5 GB） | `[实测]` 文件存在；`[日志]` 推理加载了它 |
| shape decoder | `checkpoints/ckpts/shape_dec_next_dc_f16c32_fp16.pt`，`--vae_only` | `[日志]` |
| SS 编码器（只用于做训练 latent） | `/data/lzx/TRELLIS/checkpoints/ckpts/ss_enc_conv3d_16l8_fp16`，`sample_posterior=False`，64³ | `[代码]` encode 脚本，本会话未重跑编码 |

优化：`ImageConditionedFlowMatchingCFGTrainer`，AdamW lr 1e-4，weight decay 0，EMA 0.9999，AMP bf16，`p_uncond` 0.1，logit-normal(mean 1, std 1)，`sigma_min` 1e-5，AdaptiveGradClipper(1.0, 95)，batch 4 / split 2，1 卡，`i_save=2000`。`[代码]` + `[日志]`。

### 2.3 数据

`[日志]` 启动行：

`ImageConditionedSparseStructureLatentViews: 1151 objects, 15502 view pairs, latent_frame=view, min_view_aspect=3.5`

`[代码]` `command.txt` 的 data_dir：

- `base`: `training_data/multiview_1186`
- `ss_view_latent`: `training_data/multiview_1186/ss_latents_views_canon`
- `render_cond`: `training_data/multiview_1186/renders_cond_t`

采样语义 `[代码]` `trellis2/datasets/sparse_structure_latent_views.py`：一个物体一条样本，`__getitem__` 在合格视角里随机抽一张。view 帧读 `<obj>/<iii>.npz` 的 `mean`（没有则 `z`），图按 alpha 框裁剪。`crop_scale=1.0`。

`renders_cond_t/_view_aspect.json` 存在（2026-10-12 02:58）。`_view_offaxis.json` 也存在，但时间是 **13:33 UTC，晚于训练结束 08:25**。这次训练不可能用过它。其内容本会话没有审计。

体素来源脚本 `[代码]`：`data_toolkit/generate_ss_view_voxels_canon.py`，保留目录是 `ss_view_voxels_canon_v4`。旧的绕错轴脚本 `generate_ss_view_voxels_mesh_t1.py` 仍在树里，不要再跑。

### 2.4 关键文件

| 路径 | 职责 |
|---|---|
| `configs/gen/ss_flow_lora_64_views_side_dinov3_20k.json` | **实际跑完**的配置。`latent_frame=view`，无 `canonical_latent_root`，`max_steps=20000` |
| `scripts/run_ss_lora_view_side_dinov3_20k.sh` | 实际启动脚本。`--ckpt none`，默认 GPU 7 |
| `results/..._views_side/command.txt` | 复现以这份为准 |
| `results/..._views_side/log.txt` | 逐步 loss。控制台摘要在 `/tmp/train_view_side.log`（只有 7.4 KB，头部被截过；完整进度以 `log.txt` 和本会话读到的尾部为准） |
| `configs/gen/ss_flow_lora_64_views_canon_side_dinov3_50k.json` | **没有跑出结果**的另一份。`latent_frame=canonical`，`canonical_latent_root=.../ss_latents_canon`，虽然文件名带 50k，json 里 `max_steps` 已是 20000 |
| `scripts/run_ss_lora_canon_side_dinov3_20k.sh` | 指向上面那份和输出目录 `..._views_canon_side`。该结果目录不存在 |
| `trellis2/datasets/sparse_structure_latent_views.py` | 视角过滤、view/canonical 两种目标 |
| `scripts/infer_finetuned.py` | 推理入口。加载 SS LoRA 前 `state_dict.pop('rope_phases', None)`，所以 `missing=1` 是预期 |
| `data_toolkit/encode_ss_view_latents_canon.py` | 置换后编码 |
| `nju/trellis-lora/` | TRELLIS.1 LoRA 参考实现。只读可以，不要改、不要删 |

### 2.5 实际推理命令

`[日志]` 2k 与 20k 的 `infer.log` 都加载了：对应步数的 **EMA** SS LoRA、shape decoder、shape flow 20k。没有加载 512 flow。日志里没有 canonicalize / swap_yz 的字样。与此前约定的命令一致：

```bash
cd /data/lzx/TRELLIS.2
CUDA_VISIBLE_DEVICES=5 /home/test/miniconda3/envs/trellis2/bin/python -u scripts/infer_finetuned.py \
  --image data_test/001.png --pipeline_type 1024 \
  --vae_decoder checkpoints/ckpts/shape_dec_next_dc_f16c32_fp16.pt --vae_only \
  --sparse_structure_lora results/multiview_1186_ss_flow_lora_64_views_side/ckpts/denoiser_ema0.9999_step00XXXXXX.pt \
  --ss_lora_rank 8 --ss_lora_alpha 8 \
  --flow_1024_lora results/multiview_1186_flow_1024_lora_v2/ckpts/denoiser_ema0.9999_step0020000.pt \
  --shape_lora_rank 8 --shape_lora_alpha 8 \
  --seed 42 --no_video
```

不要加 `--ss_canonicalize`，不要加 `--ss_swap_yz`，不要换 `--ss_cond_model dinov2`。

---

## 3. 当前实现状态

### 3.1 已完成且本会话核对过

- view-side 训练 20000 步结束。EMA 与非 EMA、以及 `misc_step*` 从 2000 到 20000 每 2000 步都在 `results/multiview_1186_ss_flow_lora_64_views_side/ckpts/`。
- 001、seed 42、形状 flow 固定 20k 的 GLB 已有：`results_infer/viewside_{2,10,12,14,16,18,20}k_001/001.glb`。4k/6k/8k 没有。
- 数据集支持 `latent_frame`、`min_view_aspect`、`max_view_offaxis`。后一个这次配置没开。
- 推理脚本支持 `--ss_swap_yz` 和 `--ss_canonicalize`。二者都不是当前验收配置。
- `missing=1` 的原因在推理脚本第 82–87 行：丢掉 `rope_phases`。不是权重损坏。

### 3.2 已实现但没有在当前失败上验证

- `max_view_offaxis` 过滤和 `_view_offaxis.json`。文件是训练结束后才出现的，没有用它训过，也没有在本会话统计角度分布。
- `--ss_canonicalize`。REPORT §10.3 / §11.20 说它能把最长轴摆正，但对已经崩的 A′@20k 只能部分救回，且会破坏验收。`[报告]`
- 规范帧侧视配置 `..._canon_side_...json` 与启动脚本在磁盘上，**没有对应结果目录**。不要把它写成「正在跑」或「已训完」。

### 3.3 尚未实现

- 针对「斜侧视被平均成圆管」的修复训练。代码注释里有这个担心，没有对应的成功实验。
- view-side 在 002/003/004 上的推理。
- 把 SS 和 shape flow 一起改成相机帧。TRELLIS.1 的 crrc 第二阶段配置用的仍是 `ImageConditionedSLat`（一个物体一个规范帧 latent），并不是逐视角。`[代码]` `nju/trellis-lora/configs/generation/slat_flow_img_dit_L_64l8p2_fp16_crrc_lora_bs4_split4.json`。不要写成「T1 两阶段都是相机帧，所以 T1 能工作」。

### 3.4 已知限制

- loss 会下降，同时几何变坏。本轮 0.0437 → 0.0307。A′ 上 REPORT §11.19 已有同样现象。loss 不能当验收。
- AABB 的 argmax 在斜向物体上会骗人。14k 的 argmax 仍是 z，但 y 已经 0.585、z 0.890。REPORT §11.17 对 SS occupancy 讲过同一类问题。
- 近圆截面上的 PCA 主轴角度不稳定。不要用「扭转多少度」解释 20k。`[讨论]` 此前算过又撤回，本会话没有重算 PCA。
- 001 的 `data_test/001.png` 是 RGB 黑底，不是 RGBA。用 alpha 裁前景会得到整张 518²。`[实测]`
- GPU 0–3 不要用。`003` 的 remesh 历史上可以到约 100 分钟。`[讨论]` 本会话没有重跑。
- Git：分支 `main`，HEAD `75fbf01`，相对 `origin/main` 没有提交信息显示领先/落后（`git status -sb` 只显示 `## main...origin/main`）。大量本地修改和未跟踪目录（`configs/gen/`、`scripts/`、`results`、`nju/`、`training_data` 等）。**不要提交，除非用户明确要求。不要 reset/checkout。**

磁盘本会话只量到项目顶层的一部分：`/data` 7.0 T 已用 4.3 T、剩 2.4 T（65%）。`nju/` 561 G。完整 `results/` 与 `training_data/` 的 du 被中断，没有新的分目录大小。

---

## 4. 已做实验与结论

### 4.1 本会话重新量过的 001 / seed 42

形状 flow 都是 v2 的 20k EMA。包围盒来自 trimesh 的 AABB extents，单位与 GLB 一致。长轴 = argmax。旧基线行是 `results_infer/ctrl_base_ss_datatest/001/001.glb`（2026-10-11，原厂 SS，不是旧 33k LoRA）。

| 检查点 | AABB (x, y, z) | argmax | z/y | x/y |
|---|---|---|---|---|
| 原厂 base SS | (0.2365, 0.1597, 1.0015) | z | 6.272 | 1.481 |
| view-side 2k | (0.2367, 0.1555, 1.0013) | z | 6.438 | 1.522 |
| 10k | (0.1927, 0.1548, 1.0014) | z | 6.467 | 1.244 |
| 12k | (0.1393, 0.1486, 1.0020) | z | 6.741 | 0.937 |
| 14k | (0.1325, 0.5854, 0.8896) | z | 1.520 | 0.226 |
| 16k | (0.1303, 0.8876, 0.7256) | y | 0.817 | 0.147 |
| 18k | (0.1488, 0.8499, 0.7327) | y | 0.862 | 0.175 |
| 20k | (0.1492, 0.8120, 0.6949) | y | 0.856 | 0.184 |

输入 001 前景宽高比 4.23。2k/10k 的 z/y 已经是 6.4，比照片更瘦，但和 base 的 6.27 同一量级。10k→12k 宽度相对高度从 1.24 掉到 0.94，轴还在 z。14k 不是干净的轴对调，而是 y 与 z 同时很大。16k 以后 argmax 才变成 y，截面更窄。

用户目视（`[讨论]`，不是本会话的指标）：质量从 10k 到 12k 已经明显下降；翻轴从 14k 开始；20k 的车身严重扭曲，不只是轴。AABB 与这个时间线相容，但不能代替看 mesh。

这些 GLB **没有**覆盖 002–004，也没有多种子。

### 4.2 更早的实验（REPORT，本会话未重跑）

以下只作为历史，引用前要知道它们和当前 run 不是同一配置。

| 实验 | 结果 | 依据 | 能否当当前结论 |
|---|---|---|---|
| 旧逐视角体素绕 dump 的 Z 转，和渲染相机不一致。约 71% 对象的 latent 不随视角变 | 宽高比与 latent 长轴相关 −0.492；canon v4 后 +0.971，视角不变对象 0%（30 对象 × 8 视角）。全量 1182 的校验在 §11.8 | `[报告]` §11.1–11.4、§11.8；脚本注释 `[代码]` 仍在 | 解释**旧数据**为什么坏。当前训练用的是 `ss_latents_views_canon`，不是那批坏数据 |
| 全局 y/z 交换 | 只把 29/71 的混合帧镜像成 67/33，消不掉混合 | `[报告]` §6 在 HANDOFF 里的转述，本会话未重算 IoU | 不要再提「加一次 swap 就能好」 |
| 训练 DINOv2、推理 DINOv3 | base 换 DINOv2 会塌；旧 33k 看起来稳，是条件错配把模型压回底座 | `[报告]` | 当前 run 训练和推理都是 DINOv3，这条已避开 |
| A′ `..._views_canonv2`，逐视角、不过滤宽高比 | 001 上 12k–13k 还在正确分支，**14000** 翻到 dim2，20k 四图×8 种子 0/32。loss 仍下降 | `[报告]` §11.19 | 和当前 14k 变斜的步数接近，但是**另一份配置**（无 `min_view_aspect`）。不能直接说成同一个机制已被证明。磁盘上 A′ 最新 EMA 现在是 `step0013000`，与报告里「删到 24k 仍保留」不一致，以 ls 为准 |
| B′ `..._views_canon_canonical`，规范帧、全视角 | 轴稳定，但 001 长高比约 8.0–9.2，输入 4.22。越训越瘦。用户否定轮廓 IoU | `[报告]` §11.20 被 §11.21、§11.22 纠正 | 规范帧不是当前候选。磁盘最新 EMA 是 `step0024000` |
| B′ 侧视过滤 `..._canon_side` | REPORT §11.22 说已启动。结果目录不存在。配置和脚本还在，但是另一条 `latent_frame=canonical` | `[代码]` `[实测]` | **未运行完成，没有任何指标** |
| 旧 SS LoRA `..._views` 最新 EMA `step0033000` | 历史上曾当基线；条件与坐标系后来被判为不可用 | `[报告]`，文件 `[实测]` 仍在 | 不要当产线权重 |

有效的只有：原厂 base 在 001 上轴是对的，几何偏瘦但车身还在；view-side 的 2k 接近 base。这不是 LoRA 学会了资产，是 LoRA 还没离开底座。`[实测]` AABB；「还没离开底座」是解释 `[讨论]`，本会话没有算 EMA 与底座的参数残差。

无效的，且有证据：

- 把逐视角目标训到 LoRA 真正主导（A′@20k `[报告]`；当前 16k–20k `[实测]`）。轴和/或车身坏掉。
- 规范帧全视角训久（B′@24k `[报告]` + 用户目视）。轴对，比例更差。
- 用轮廓 IoU、二面角、loss、或只看 argmax 来排名。报告和这次 14k 的 AABB 都表明它们会排错。

### 4.3 还不能下结论的

- 当前 15502 对里，宽高比 ≥ 3.5 的斜侧视（30°/45°）占多少，以及它们的 latent 长轴是否落在 index1 与 index2 之间。数据集注释 `[代码]` 第 26–29 行写了这个风险：「那些图看起来都像长车，目标朝向却差几十度，模型会把车身平均成圆管。」这与 12k 变瘦、14k 起 yz 同时变大 **相容**，但是分布没有数过。**不要写成已证实根因。**
- `0.9999^n` 等于「还剩多少底座能力」。没验证。
- 「rank 8 表达不了刚体旋转」。没验证。
- 20k 的截面是不是几何上的圆管。此前用切片宽高比得到约 1.00–1.05，随后认定近圆截面的 PCA 角度不可靠。本会话没有重做切片。可以引用为未复核的旧测量，不能当新证据。
- canon 数据现在是否每张图都配对自己的体素。报告说修好了；本会话没有抽查 001 的相机、体素和 latent。

---

## 5. 重要设计决策

### 5.1 已经定了的

- 验收种子 42，四张测试图，长度轴 z，还要看比例；不开 canonicalize。`[报告]` 记录的用户指示。
- 条件必须是 DINOv3@512、`crop_scale=1.0`，和推理预处理一致。当前配置遵守了这一点。`[代码]`
- 新 run 用 `renders_cond_t`，不用混有白底不透明图的 `renders_cond/`。`[报告]` HANDOFF §6.1-10；当前 command.txt 确实是 `renders_cond_t`。
- 从原厂 SS 权重开始，`--ckpt none`，不从 B′ 或 A′ 续。当前 command.txt 如此。
- 形状 flow 在这轮诊断里冻结在 v2 20k，用来隔离 SS 检查点。`[日志]`
- 编码置换用 `(2,0,1)`，不用一次 y/z 对换（对换行列式为 −1，是镜像）。`[代码]` 脚本注释；搜索脚本的 IoU 数字是 `[报告]`，本会话未重跑搜索。

### 5.2 讨论过但没决定

- 下一步是收紧偏角（`max_view_offaxis`），还是放弃逐视角、改别的监督。没有新实验。
- 是否重训 shape flow。当前失败在 SS 检查点更换时出现，形状权重没变，所以优先还是 SS 目标；不能据此证明 shape flow 没有自己的问题。
- 旧检查点和中间体素能不能删。用户曾要求清理多余文件，盘点被中断，**没有删任何东西**。

### 5.3 不要轻易推翻

- 不要改 `nju/`。
- 不要 `git reset` / `git checkout` / 提交，除非用户明确说。
- 不要占用 GPU 0–3。
- 不要用 `--ss_canonicalize` 或 `--ss_swap_yz` 生成「验收通过」的图。
- 不要把 REPORT §11.20 的「B′@24k 最好」或 §11.22 的「canon_side 正在训练」当成现状。
- 不要再跑 `generate_ss_view_voxels_mesh_t1.py` 去覆盖 `ss_latents_views_canon`。
- 训练快照默认不可靠：SS 数据集 `skip_snapshot` 默认 True。`[代码]` 数据集注释。`samples/` 里即使有图，也不等于线上 GLB。

---

## 6. 当前最重要的问题

核心阻塞：唯一训完的新 SS LoRA（view-side）在 001 上，12k 已变瘦，14k 起车身斜在 yz 里，16k 后长轴变成 y。20k 不能用。2k 看起来正常，是因为它还接近原厂底座，不是这套目标已经学对。

已经试过、并且有证据表明不够：

1. 修体素化的环绕轴并做 `(2,0,1)`，得到现在的 `ss_latents_views_canon`。这修的是旧数据图文无关，不是这轮 14k 翻轴。
2. 全视角逐视角（A′）：约 14k 翻到错误轴。
3. 全视角规范帧（B′）：轴稳定，比例训得更瘦，用户否定。
4. 只留宽高比 ≥ 3.5 的**逐视角**（本轮）：轴仍然在 14k–16k 离开 z，而且更早出现变瘦。侧视过滤没有解决逐视角目标的问题。
5. 规范帧 + 同样的宽高比过滤：只写了配置和脚本，没有结果。

仍然难的地方：宽高比 ≥ 3.5 并不等于正侧视。代码自己写了 30°/45° 会留下来。`max_view_offaxis` 没开，角度表还是训完之后才生成的。所以「斜视被平均」是当前最吻合现象的假设，缺一张实际的角度直方图和若干样本的图/latent 对照。

最值得优先做的（假设，先验证再训练）：

1. 读 `_view_offaxis.json` 的定义（哪个脚本写的、角度相对谁）。本会话没读。
2. 在**实际参加训练的** 15502 对上统计偏角，而不是在全部 24 视角上统计。确认 ≥3.5 的集合里有多少超过 15° 或 20°。
3. 抽 001 的合格视角：渲染图、体素长轴、latent 解码后的三轴跨度。确认不是又一次固定的轴置换错误。纯侧视目标在置换后应该长在 index1；若正侧视样本已经长在别的轴，问题就回到数据，而不是斜视平均。
4. 只有 2 和 3 都支持「正侧视目标是对的、斜视很多」时，才用现有数据集参数 `max_view_offaxis` 开一个短训（仍从原厂权重，`--ckpt none`，不要续 20k）。先看 001 的 2k/4k/6k GLB 是否还在变瘦，不要直接训满 20k。
5. 不要并行再开规范帧。B′ 的瘦管已经有测量。

---

## 7. 下一步行动

1. 读 `trellis2/datasets/sparse_structure_latent_views.py` 的 `_filter_offaxis`，以及生成 `_view_offaxis.json` 的脚本（在 `scripts/utils/` 或 `data_toolkit/` 里搜文件名）。确认角度定义。不要假设 json 的单位和零点。
2. 用与训练相同的过滤（美学分、`cond` 图存在、aspect ≥ 3.5、最多 24 视角）统计偏角直方图，写出超过 10°/20°/30° 的比例。这是读数据，不要开 GPU 训练。
3. 对物体 001（以及如果便宜的话再加一个）导出每个合格 view 的：图路径、偏角、体素三轴跨度、是否存在 `ss_latents_views_canon/<obj>/<iii>.npz`。
4. 若正侧视样本的长度不在 index1：停止，回到 `generate_ss_view_voxels_canon.py` 与编码置换，不要开新训练。
5. 若正侧视是对的且斜视占比高：改一份新 json（不要覆盖已经跑完的 `ss_flow_lora_64_views_side_dinov3_20k.json`），设 `max_view_offaxis`，`max_steps` 先小，`latent_frame` 仍为 `view`，其余与本轮相同。输出目录用新名字。
6. 推理仍用形状 flow 20k、seed 42、无 canonicalize。先只跑 001。比较 x/y 与 z/y 和本文件第 4.1 节的表。用户要看 mesh，不要只交 AABB。
7. 001 在短训里不再变瘦、轴仍为 z 之后，再补 002–004。`003` remesh 可能很慢。
8. 在那之前不要更新「产线检查点」。2k 可以当「接近 base」的对照，不能当交付权重。
9. 清理磁盘另说。见第 8 节。本交接没有授权删除。

---

## 8. 新会话注意事项

必须先看：

- 本文件。
- `results/multiview_1186_ss_flow_lora_64_views_side/command.txt` 和同目录 `config.json`。
- `configs/gen/ss_flow_lora_64_views_side_dinov3_20k.json`。
- `trellis2/datasets/sparse_structure_latent_views.py` 前 170 行。
- `results_infer/viewside_12k_001/infer.log` 与 `viewside_20k_001/infer.log` 的权重加载段。
- REPORT 只读 §11.16–§11.22，并且记住 §11.22 描述的 run **没有产物**。

不能直接相信：

- `HANDOFF.md` §6.3（写着 B′ 在跑、B′@24k 最好）。
- REPORT §11.20 的排名和「IoU 单调变好」。
- REPORT §11.22 的「`..._canon_side` 已在 GPU7 训练」。目录不存在；真正训完的是 `..._views_side`，`latent_frame=view`。
- 任何「A′ 检查点保留到 24k/28k」。今天 `ls` 到的最新 EMA 是 `step0013000`。
- 对话里关于 PCA 扭转角、EMA 残差百分比、rank 8 能力的句子。
- Hindsight / 记忆工具。本环境调用返回 401，没有知识页。

有风险、未确认不能做：

- 删除 `nju/`、`checkpoints/`、`ss_latents_views_canon`、`ss_view_voxels_canon_v4`、`renders_cond_t`、view-side 的 ckpt、shape flow 20k。
- `rm` 由 SSH 工具拦截；审批在本会话是关闭的，失败后不要改用 Python 删，也不要加 `sandbox_permissions`。
- 覆盖 `data_test/` 或训练数据。
- 在 GPU 0–3 上启动训练或推理。
- 从 20k 续训。那会把已经坏的目标继续学下去。
- 把 `results` 软链接和 `/data/TRELLIS.2/results` 当成两份数据。HANDOFF 曾写仓库 `results` 链到 `/data/TRELLIS.2/results`。本会话 `ls` 看到的是 `/data/lzx/TRELLIS.2/results/...`。删之前先 `readlink`。

环境：服务器上一次成功命令是 `test@192.168.65.39`，家目录 `/home/test`。项目路径用 `/data/lzx/TRELLIS.2`。Blender 旧文档写过 `/data/lzx/opt/blender-4.5.1-linux-x64/blender`，本会话没验证二进制还在不在。

---

## 9. 交接信息的可信度

| 结论 | 依据 |
|---|---|
| 跑完的是 view 帧、aspect≥3.5、DINOv3、20k、从原厂权重、1151/15502 | 配置、`command.txt`、训练日志启动行、`/tmp/train_view_side.log` 尾部「Training finished」 |
| 优化器、LoRA 目标、150 层、可训练参数量 | 训练日志 |
| 推理用 EMA SS + shape 20k，`missing=1` 来自弹出 `rope_phases` | `infer.log` + `scripts/infer_finetuned.py` |
| 001 各步 AABB、argmax、z/y、x/y | 本会话 trimesh，2026-10-12 |
| 001–004 前景宽高比 4.23 / 3.22 / 3.93 / 5.69 | 本会话，`max(RGB)>16` |
| 14k 已斜、16k 才 argmax=y；用户认为 12k 形状已坏、20k 严重扭曲 | AABB 是实测；「严重扭曲」是用户目视，没有新的形状指标 |
| canon_side 没跑 | 无结果目录；脚本和 json 仍指向它 |
| `_view_offaxis.json` 未参与训练 | 文件 mtime 13:33 > 训练结束 08:25 |
| A′ 14k 翻转、B′ 过瘦、旧体素轴错误、验收标准条文 | REPORT，未重跑 |
| 斜视平均是这轮失败的原因 | **未验证**。只有代码注释和 AABB 时间线 |
| 2k「好」是因为还近底座 | **未验证**。没有参数残差 |
| GPU 空闲情况、无 train/infer 进程 | 本会话 `nvidia-smi` 与 `ps`，写完文档后可能变化 |
| Git 脏、HEAD `75fbf01` | 本会话 `git status` / `git log` |
| 磁盘分目录大小、哪些文件可删 | **未完成**。不要根据旧 HANDOFF 的「可删 v2/v3 体素」动手，除非重新 `ls` 确认那些目录还在且不是 v4 |

---

## 附：和 TRELLIS.1 LoRA 的差别（避免再误移植）

T1 参考配置 `[代码]`：`nju/trellis-lora/configs/generation/ss_flow_img_dit_L_16l8_fp16_views_lora_bs4_split2.json`。数据集 `nju/trellis-lora/trellis/datasets/sparse_structure_latent_views.py`。

相同：都是官方「多视图 → 一个规范帧」底座上的 LoRA；rank/alpha/插层、lr、EMA、batch 4/split 2、logit-normal、`p_uncond` 相同。T1 第二阶段 crrc 配置也不是逐视角 SLat。

不同，且和这次失败有关的：

- T1 的逐视角体素是渲染目录里的 `{view:03d}_voxel.ply`，和那张图同一次管线。T2 是后写的 `generate_ss_view_voxels_*.py`，第一版绕错轴。
- T1 留 12 个视角，不过滤宽高比，DINOv2@518，裁剪 1.2，底座是 L（1024 通道、24 层、APE）。
- 当前 T2 只留 aspect≥3.5，丢掉端视、留下看起来都像长车的斜侧视，DINOv3@512，裁剪 1.0，底座 1.3B RoPE。条件模型和裁剪是按 T2 推理改的，这两项本身不是 bug。

不要把「T1 能用，所以逐视角配方没问题」写成事实。本会话没有重跑 T1，也没有 T1 在这四张图上的 GLB。

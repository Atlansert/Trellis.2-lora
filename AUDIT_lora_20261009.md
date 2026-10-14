# LoRA 微调方案审计报告

**日期**：2026-10-09
**触发问题**：生成模型"细节不足、边缘流畅度不够"
**审计对象**：三条 flow 链路（SS Stage1 / Shape Stage2 / Tex flow）的 LoRA 微调方案
**结论**：**是训练方案的问题，但不是超参数问题，而是"训练条件编码器 ≠ 推理条件编码器"的配置性缺陷，且缺陷只发生在 Stage 1。**

---

## 1. 决定性证据（实测，非推断）

用同一噪声种子、同一 LoRA 权重，仅替换条件编码器，对比输出 occupancy 与训练目标（per-view SS latent）的一致性。
评测集：6 个对象的视角 0；指标：64³ 栅格上的 IoU。

| 条件 | 预测体素数 | 真值体素数 | IoU | 漏检率 | 多余率 |
|---|---|---|---|---|---|
| A 预训练 + DINOv3@512（推理实际路径） | 5044 | 2568 | **0.066** | 91.4% | 95.0% |
| B LoRA + DINOv3@512（**线上实际跑的**） | 2782 | 2568 | **0.054** | 92.1% | 92.1% |
| C LoRA + DINOv2@518（**训练时的条件**） | 2719 | 2568 | **0.525** | 41.4% | 45.2% |
| G 预训练 + DINOv2@518 | 12377 | 2568 | 0.017 | — | — |
| D LoRA + DINOv2 + 50 步 | 2398 | 2568 | 0.405 | 47.1% | 49.7% |
| E LoRA + DINOv3 + 50 步 | 2655 | 2568 | 0.030 | 93.6% | 94.4% |
| F LoRA + DINOv3 + 训练裁剪 | 10714 | 2568 | 0.018 | 93.9% | 96.9% |

**读法**：

1. **换条件编码器，IoU 从 0.054 跳到 0.525，差 10 倍。** 同一权重、同一噪声种子、同一采样器，唯一变量是 cond 张量来自哪个编码器。这是根因，不需要再猜。
2. **LoRA 在推理时几乎不起作用。** 同噪声种子下输出间的 IoU：

   | | A 预训练+DINOv3 | B LoRA+DINOv3 | C LoRA+DINOv2 |
   |---|---|---|---|
   | A 预训练+DINOv3 | 1.000 | 0.470 | 0.066 |
   | B LoRA+DINOv3 | 0.470 | 1.000 | 0.064 |
   | C LoRA+DINOv2 | 0.066 | 0.064 | 1.000 |

   - DINOv3 条件下：预训练 vs LoRA 差异 **53.0%**
   - DINOv2 条件下：预训练 vs LoRA 差异 **98.1%**

   也就是说，线上推理时 LoRA 只把输出挪动了不到一半，且挪动方向不是真值方向（B 的 IoU 0.054 甚至略低于 A 的 0.066）。**名义上用着微调权重，实际行为大体退回预训练**——这正好对应"细节不足"。
3. **步数不是主因。** E（50 步）0.030 vs B（12 步）0.054——在条件已经错的前提下，把采样步数从 12 提到 50 没有救命。**先别急着改采样步数。**
4. **裁剪尺度也不是主因。** F（用训练的 1.2× 裁剪喂 DINOv3）反而恶化到 0.018。

---

> **实测更正（2026-10-10）**：条件编码器输出 token 数经实测为
> `DinoV2LocalFeatureExtractor@518` → **1297**，`DinoV3FeatureExtractor@512` → **1029**
> （两者 hidden 均为 1024）。此前文中 1370 / 1025 为估算值，结论不受影响。

## 2. 根因定位

| 链路 | 训练时 cond 编码器 | 推理时 cond 编码器 | 是否一致 |
|---|---|---|---|
| SS Stage1 | `DinoV2LocalFeatureExtractor` @518（hidden 1024，实测 1297 token） | `DinoV3FeatureExtractor` @512（patch 16，实测 1029 token） | ❌ **不一致** |
| Shape Stage2 | `DinoV3FeatureExtractor` @1024 | `DinoV3FeatureExtractor` @1024 | ✅ 一致 |
| Tex flow | `DinoV3FeatureExtractor` @1024 | `DinoV3FeatureExtractor` @1024 | ✅ 一致 |

- 训练侧：`configs/gen/ss_flow_lora_64_views_train_to_1000k.json` → `DinoV2LocalFeatureExtractor(model_name=/data/lzx/ReconViaGen/checkpoints_dinov2, image_size=518)`（`results/multiview_1186_ss_flow_lora_64_views/config.json` 已确认实际运行值）。
- 推理侧：`checkpoints/pipeline.json` → `image_cond_model = DinoV3FeatureExtractor(/data/lzx/TRELLIS.2/dinov3)`，`run()` 内部 `get_cond([image], 512)` 喂给 SS。
- **上游官方 SS 配方用的是 DINOv3@512**（`configs/gen/ss_flow_img_dit_1_3B_64_bf16.json` 已核对）。所以这是本地在 SS 上单方面偏离上游、而没有同步改推理侧的后果。

为什么 G（预训练+DINOv2）会产生 12377 个体素（真值 2568）？因为 DINOv2 特征对预训练模型本身就是 OOD，输出崩坏。而 LoRA 恰好**是在这个 OOD 条件下训练的**——它学的是"把 DINOv2 特征翻译回正确结构"。C 的 0.525 证明它学得很好。问题在于线上根本不给它 DINOv2 特征。

---

## 3. 次要问题（真实存在，但不是本次边沿问题的原因）

| # | 问题 | 证据 | 影响 |
|---|---|---|---|
| 1 | SS 过训练 | 28k–33k 为最优窗口，300k+ 破洞（HANDOFF §6.1） | 已有认知，别用晚期 ckpt |
| 2 | `real_view_prob 0.75` 基本失效 | `is_real` 仅存在于 505/1183 对象、518/28886 帧（1.8%）；抽样对象完全没有 | 该超参形同虚设 |
| 3 | `max_tokens 32768` 从未生效 | shape_1024 实际 token 最大 5565（p50 2386）；metadata 门槛用的是 512 尺度的 `shape_latent_tokens`（p50 630） | 不会截断，无实际危害 |
| 4 | tex 归一化常量漂移 | `pbr_slat_normalization.std[16]`：训练 `2.292295` vs 推理 `2.295916`（Δ 0.0036）；其余向量完全一致 | 单通道细微色偏，需修 |
| 5 | `full_pbr=true` 砍掉样本 | 1182 → 810 | 有意为之 |
| 6 | `aesthetic_score` 是占位值 | 全部 5.50，而门槛 4.5 | 该过滤条件永远通过，无实际筛选力 |
| 7 | SS ckpt 家族不一致 | 14/23 个 EMA ckpt 带 `rope_phases`（早期是 complex64，后期是 bf16）；`infer_finetuned.py` 已 pop 掉 | 已有防护，安全 |
| 8 | 训练快照采样配置 ≠ 推理采样配置 | 快照 steps=50/guidance 3.0；推理 steps=12/guidance 7.5 | 快照图不能代表线上效果 |
| 9 | 历史 detail/geometry 配方已废弃 | rank16 + 含 `adaLN_modulation.1`/`t_embedder.mlp.*` 目标 + uniform t + ema0.999 + lr5e-5 + max_steps 12000 | 当前 v2 是 rank8/alpha8，共 5 个 attention 目标 |

注意 #8 与实验 1.3 相互印证：**快照图看起来还行，是因为快照用的是 50 步 + guidance 3.0；线上跑的是 12 步 + guidance 7.5。** 这也是为什么问题直到出 GLB 才暴露。

---

## 4. 建议的修复路径

**优先级 1 —— 让 SS 的条件编码器对齐（二选一）**

- **方案 A（推荐，改动小、当天可验证）**：让推理时 SS 走 DINOv2@518，shape/tex 仍走 DINOv3@1024。
  障碍：`pipeline.json` 只有一个全局 `image_cond_model`，三条链路共用。需在 `trellis2_image_to_3d.py` 里为 SS 增加独立编码器（如 `self.ss_cond_model`），或加一个 `DinoV2LocalFeatureExtractor` 作为第二编码器，`run()` 中 SS 用 518、后续用 1024。
- **方案 B（更正统，但要重训）**：改用 DINOv3@512 重新提取 SS 训练条件并重训 SS LoRA。数据管线已具备（64³ voxel → encode latent 已就绪），只需换 cond model 重跑 `ss_flow_lora_64_views`。好处是三条链路统一到 DINOv3，消除隐患。
- **可以立刻做的验证**：按上面方案 A 接好编码器后，跑一次 6 张图的推理，看边缘是否恢复——因为实验已经证明 C 条件能让 SS 输出回到 IoU 0.525。

**优先级 2 —— 重建一个诚实的评测基线**

训练快照的采样参数与线上不一致，导致"看到的"和"得到的"不是一回事。建议把快照采样参数对齐线上（steps=12、guidance 7.5、rescale 0.5、interval [0.6,1.0]、rescale_t 3.0），这样 i_sample 出来的图才可信。

**优先级 3 —— 顺手修的小问题**

- `pipeline.json` 的 `pbr_slat_normalization.std[16]` 对齐训练值 `2.292295`
- `real_view_prob` 若仍需要，先补 `is_real` 标注；否则从 config 里删掉以免误读
- 清理 `max_tokens 32768` 之类的无效超参

**不要动的东西**：优化器/学习率/EMA/t 调度这几项三条链路一致且与上游一致，实验也证明它们不是本次问题的来源。**在条件编码器对齐、并拿到可信基线之前，不要调任何训练超参数。**

---

## 5. 审计方法与复现

实验脚本：`/tmp/ss_ab6.py`（结果日志 `/tmp/ss_ab6.log`）。
做法：加载 `checkpoints` 管线 → 注入 SS LoRA（rank8/alpha8，5 个 attention 目标，`rope_phases` 已 pop）→ 对每个样本分别用 DINOv3@512 与 DINOv2@518 生成 cond → 同一 `torch.manual_seed(42)` 采样 → 用 `sparse_structure_decoder` 解码真值 latent 得 occupancy → 计算 IoU / 漏检 / 多余。

**已知局限**（如实说明）：
- 评测基于 SS latent 的 occupancy IoU，不是最终 GLB 的视觉评分。要下最终结论仍需按方案 A 跑一次端到端 GLB 对比。
- 真值 latent 本身经过 `voxelize(64³) → encode → decode` 往返，所以 C 的 0.525 里含一部分 VAE 往返损失；但 A/B/C/F 共用同一真值，横向对比有效。
- 只测了 6 个样本的视角 0（侧视图），未覆盖全部视角分布。

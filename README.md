# Trellis.2-lora

TRELLIS.2 的**代码主干镜像**，包含本项目在其上做的 **LoRA 微调**改造。

- 上游项目：<https://github.com/microsoft/TRELLIS.2>
- **代码结构说明（建议先读）**：[`PROJECT_CODE_GUIDE.md`](PROJECT_CODE_GUIDE.md)
- 授权：见 [`LICENSE`](LICENSE)

## 仓库内容

| 路径 | 说明 |
|---|---|
| `trellis2/` | 核心库：模型、稀疏算子、训练器、数据集、推理流水线、表征与渲染 |
| `scripts/` | 实验脚本层：微调推理入口、推理/验收契约、历史训练与推理脚本 |
| `configs/` | 训练与模型配置（SS / shape / texture / VAE / LoRA） |
| `data_toolkit/` | 数据工具链：资产准备、Blender 渲染、latent 编码、坐标契约 |
| `tests/` | CPU 单元测试（契约、检查点、验收、LoRA 幂等性） |
| `train.py` / `run_trellis.2.py` | 训练与推理入口 |

## 未包含（因此本仓库不能直接跑通）

数据集、模型权重、训练产物与实验证据，合计数百 GB：

`nju/`、`results/`、`results_infer/`、`training_data/`、`data_for_train/`、`new_data*/`、`checkpoints/`、
`trellis2_finetune_release/`、`rmbg/`(权重)、`dinov3/`(权重)、`o-voxel/`、`assets/`、`diagnostics/`(实验产物)。

需要按上游说明自行准备权重与数据后才能运行。

## 说明

- 代码中已加入精简中文注释，便于阅读。
- 本项目规则：`nju/` 只读不改；训练/推理不使用 GPU0–3。

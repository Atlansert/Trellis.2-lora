# Trellis.2-lora — 代码镜像说明

本仓库是 TRELLIS.2 项目的**代码镜像**（含本项目的 LoRA 微调改动），用于阅读与协作。

- 上游项目：https://github.com/microsoft/TRELLIS.2
- 本仓库**只包含源代码/配置/文档**，不包含数据集、模型权重、训练产物与实验证据（这些体积达数百 GB）。
- 未包含：`nju/`、`results/`、`results_infer/`、`training_data/`、`data_for_train/`、`new_data*/`、`checkpoints/`、`rmbg/`(权重)、`dinov3/`(权重)、`trellis2_finetune_release/`。
- 因此本仓库**不能直接跑通**完整推理/训练，需按上游说明自行准备权重与数据。
- 代码说明见 `PROJECT_CODE_GUIDE.md`。
- 第三方子目录（`o-voxel/third_party/eigen` 等）为上游子模块，未纳入。

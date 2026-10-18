#!/usr/bin/env bash
# ── 中文摘要 ─────────────────────────────────────────────
# 【已废弃 · DEPRECATED 2026-10-17】请改用 scripts/run_tex_flow_lora_pool1302.sh。
#
# 为什么废弃：本脚本指向 training_data/multiview_1186 与 --ckpt latest。该数据集已经不存在
# （旧数据清理时被删），所以脚本一定跑不起来；tex 训练的数据入口已经换成 pool1302。
#
# 替代：bash scripts/run_tex_flow_lora_pool1302.sh [GPU号，默认6]
#   - 配置 configs/gen/tex_flow_lora_1024_pool1302.json（= 官方 tex 配方，只改 max_steps 30000 / i_save 2000）
#   - 数据 training_data/pool1302/train（tex 口径 817 对象）
#   - 输出 results/pool1302_tex_flow_lora_1024/，--ckpt none 全新训练
#
# 保留本文件是为了让所有指向旧脚本的文档/脚本立刻拿到这条提示，而不是静默地用错数据训练。
# [DEPRECATED] Texture SLat Flow (1024) LoRA — old multiview_1186 launcher,
# superseded by scripts/run_tex_flow_lora_pool1302.sh. The dataset it points at
# was deleted, so this launcher can no longer run. It refuses to start on purpose.
set -euo pipefail
cat >&2 <<'EOF'
================================================================================
[DEPRECATED] scripts/run_tex_flow_lora.sh 已废弃，不会启动任何训练。

原因：它指向的 training_data/multiview_1186 已被删除（旧数据清理），
      且用 --ckpt latest 续训旧输出，与 pool1302 数据口径不符。

请改用：
    bash scripts/run_tex_flow_lora_pool1302.sh [GPU号，默认6]

新脚本：configs/gen/tex_flow_lora_1024_pool1302.json + training_data/pool1302/train
        （tex 口径 817 对象）-> results/pool1302_tex_flow_lora_1024/
================================================================================
EOF
exit 64
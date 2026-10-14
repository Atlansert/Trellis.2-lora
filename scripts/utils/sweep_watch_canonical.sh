#!/usr/bin/env bash
# ── 中文摘要 ─────────────────────────────────────────────
# 用途：守候型 watcher：等"规范帧版"SS LoRA 的 10k..50k 检查点出现后逐个跑扫描推理（shape 固定 20k）。
# 用法：bash scripts/utils/sweep_watch_canonical.sh [gpu，默认 7]；内部轮询 sleep 120，可长时间常驻。
# 输出：results_infer/_sweep_canonical.log + 各档结果目录；会 mkdir results_infer。
# 边界：只读检查点并调用 infer_ss_sweep_generic.sh（GPU 推理）；不训练、不改权重。
# 等"规范帧版"SS LoRA 的检查点出现后逐个扫描（10k..50k，shape 固定 20k）。
# Usage: sweep_watch_canonical.sh <gpu>
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd -P)"
GPU="${1:-7}"
CKDIR="$ROOT/results/multiview_1186_ss_flow_lora_64_views_canonical/ckpts"
LOG="$ROOT/results_infer/_sweep_canonical.log"
mkdir -p "$ROOT/results_infer"

echo "[$(date -Is)] canonical sweep watcher 启动 GPU=$GPU" | tee -a "$LOG"
for S in 10000 20000 30000 40000 50000; do
    CKPT="$CKDIR/denoiser_ema0.9999_step$(printf '%07d' "$S").pt"
    waited=0
    while [[ ! -f "$CKPT" ]]; do
        sleep 120; waited=$((waited+120))
        [[ $((waited % 3600)) -eq 0 ]] && echo "[$(date -Is)] 等 canonical step $S（$((waited/60)) 分钟）" >> "$LOG"
    done
    echo "[$(date -Is)] canonical step=$S 就绪，开始推理" | tee -a "$LOG"
    bash "$ROOT/scripts/infer_ss_sweep_generic.sh" "$CKDIR" "$S" "$GPU" "ss_canonical" >> "$LOG" 2>&1
    echo "[$(date -Is)] canonical step=$S 结束" | tee -a "$LOG"
done
echo "[$(date -Is)] canonical 全部完成" | tee -a "$LOG"
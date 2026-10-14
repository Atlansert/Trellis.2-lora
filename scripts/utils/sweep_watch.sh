#!/usr/bin/env bash
# 等待 SS 检查点出现后，逐个执行 task 4 的推理扫描（10k..50k，每 10k 一档）。
#
# Usage: $0 <gpu>
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd -P)"
GPU="${1:-7}"
CKPT_DIR="$ROOT/results/multiview_1186_ss_flow_lora_64_views_dinov3_512/ckpts"
LOG="$ROOT/results_infer/_sweep_watch.log"
mkdir -p "$ROOT/results_infer"

echo "[$(date -Is)] watcher 启动，GPU=$GPU" | tee -a "$LOG"
for S in 10000 20000 30000 40000 50000; do
    CKPT="$CKPT_DIR/denoiser_ema0.9999_step$(printf '%07d' "$S").pt"
    waited=0
    while [[ ! -f "$CKPT" ]]; do
        sleep 120
        waited=$((waited + 120))
        if [[ $((waited % 1800)) -eq 0 ]]; then
            echo "[$(date -Is)] 仍在等待 step $S（已等 $((waited/60)) 分钟）" | tee -a "$LOG"
        fi
    done
    echo "[$(date -Is)] 检查点就绪 step=$S，开始推理" | tee -a "$LOG"
    bash "$ROOT/scripts/infer_ss_dinov3_sweep_gpu.sh" "$S" "$GPU" >> "$LOG" 2>&1
    echo "[$(date -Is)] step=$S 推理结束 rc=$?" | tee -a "$LOG"
done
echo "[$(date -Is)] 全部 5 档完成" | tee -a "$LOG"
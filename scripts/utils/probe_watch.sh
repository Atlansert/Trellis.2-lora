#!/usr/bin/env bash
# 帧归属判定 watcher：<run> ∈ {A: 视角帧监督, B: 规范帧监督}
#
# 修正：原来一个脚本先跑完 A 再跑 B，导致 B 的早期检查点（2k/5k）被 A 的 20k..50k
# 阻塞数小时。改为按 run 分开启动，互不阻塞。
#
# Usage: probe_watch.sh <A|B> <gpu>
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd -P)"
PY="/home/test/miniconda3/envs/trellis2/bin/python"
RUN="$1"; GPU="${2:-6}"
case "$RUN" in
  A) CKDIR="$ROOT/results/multiview_1186_ss_flow_lora_64_views_dinov3_512/ckpts"
     STEPS="20000 30000 40000 50000"; TAGP="A_view" ;;
  B) CKDIR="$ROOT/results/multiview_1186_ss_flow_lora_64_views_canonical/ckpts"
     STEPS="2000 5000 10000 20000 50000"; TAGP="B_canon" ;;
  *) echo "用法: $0 <A|B> <gpu>"; exit 1 ;;
esac
LOG="$ROOT/results_infer/_frame_probe_${RUN}.log"
mkdir -p "$ROOT/results_infer"
export CUDA_VISIBLE_DEVICES="$GPU"
cd "$ROOT"
echo "[$(date -Is)] probe[$RUN] 启动 GPU=$GPU 步骤=$STEPS" | tee -a "$LOG"

for S in $STEPS; do
    CKPT="$CKDIR/denoiser_ema0.9999_step$(printf '%07d' "$S").pt"
    waited=0
    while [[ ! -f "$CKPT" ]]; do
        sleep 90; waited=$((waited+90))
        [[ $((waited % 3600)) -eq 0 ]] && echo "[$(date -Is)] 等 $RUN step $S（$((waited/60)) 分钟）" >> "$LOG"
    done
    echo "[$(date -Is)] === probe $RUN step $S ===" >> "$LOG"
    "$PY" -u scripts/utils/ss_frame_probe.py --tag "${TAGP}_step$S" --cond dinov3 --n 6 \
        --ckpt "$CKPT" >> "$LOG" 2>&1
done
echo "[$(date -Is)] probe[$RUN] 全部完成" | tee -a "$LOG"
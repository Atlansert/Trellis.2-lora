#!/usr/bin/env bash
# ── 中文摘要 ─────────────────────────────────────────────
# 用途：Texture SLat Flow (1024) LoRA 微调，数据 = training_data/pool1302/train（tex 口径 817 对象）。
# 用法：bash scripts/run_tex_flow_lora_pool1302.sh [GPU号，默认6] [额外 train.py 参数...]
#       例：bash scripts/run_tex_flow_lora_pool1302.sh 6 --tryrun   # 只建模型/dataset，不跑真训练
#       --ckpt none 表示全新训练；本脚本不续训（改过配置请换新输出目录，避免混历史）。
# 配方：与官方 configs/gen/tex_flow_lora_1024_train.json 逐键一致，只改 max_steps 300000→30000、i_save 10000→2000
#       （对应已验收的 shape flow LoRA 30k 预算）；r8/a8、只挂 5 个 attention 目标、EMA0.9999、lr1e-4。
# 底模：checkpoints/ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.safetensors（官方 imgshape2tex 1.3B 1024）。
# 数据：pbr_latents/tex_enc_next_dc_f16c32_fp16_1024（930 个纹理 latent，坐标与 shape latent 逐 token 相等）
#       + shape_latents/shape_enc_next_dc_f16c32_fp16_1024 + renders_cond_t。
# 输出：results/pool1302_tex_flow_lora_1024/（config.json / command.txt / log.txt / ckpts / samples）。
# 边界：长跑 GPU 训练；只读 training_data/pool1302/、configs/gen/、checkpoints/；只写上面那个输出目录。
#       不要用 GPU 0-3（他人 vLLM）或 4/5（常驻 ComfyUI），本脚本默认 6。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
PY="/home/test/miniconda3/envs/trellis2/bin/python"
GPU="${1:-6}"
# 第 2 个及以后的参数原样透传给 train.py（如 --tryrun / --profile），默认调用不受影响
EXTRA=("${@:2}")
OUT="$ROOT/results/pool1302_tex_flow_lora_1024"
DATA='{"train_data":{"base":"./training_data/pool1302/train","pbr_latent":"./training_data/pool1302/train/pbr_latents/tex_enc_next_dc_f16c32_fp16_1024","shape_latent":"./training_data/pool1302/train/shape_latents/shape_enc_next_dc_f16c32_fp16_1024","render_cond":"./training_data/pool1302/train/renders_cond_t"}}'

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$ROOT"
exec "$PY" -u train.py \
  --config configs/gen/tex_flow_lora_1024_pool1302.json \
  --output_dir "$OUT" \
  --ckpt none \
  --num_gpus 1 \
  --data_dir "$DATA" \
  --auto_retry 3 \
  "${EXTRA[@]}"

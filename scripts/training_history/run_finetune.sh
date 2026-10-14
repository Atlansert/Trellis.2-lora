#!/usr/bin/env bash
# =============================================================================
# TRELLIS.2 迭代微调 — 统一脚本
# =============================================================================
# 自动识别数据格式，支持接力训练。
#
# 数据目录格式 (自动检测):
#   【Flat】 路径下直接是 .glb/.obj 文件 (+ 可选同名图片)
#     data/
#       ├── model_001.glb
#       ├── model_001.png       (可选)
#       └── model_002.glb
#
#   【子目录】每个子目录含恰好一个 .glb/.obj [+ 图片]
#     data/
#       ├── sample_001/
#       │   ├── model.glb
#       │   └── image.png       (可选)
#       └── sample_002/
#           └── model.glb
#
# 接力规则:
#   默认从 results/latest_vae/ 和 results/latest_flow/ 继续
#   --fresh 强制从 checkpoints/ckpts/ 预训练权重开始
#
# 用法:
#   bash scripts/run_finetune.sh --data_dir ./my_data
#   bash scripts/run_finetune.sh --data_dir ./my_data --fresh
#   bash scripts/run_finetune.sh --data_dir ./my_data --name my_exp --skip-flow
# =============================================================================

set -e

# =============================================================================
# 参数解析
# =============================================================================
SKIP_VAE=false
SKIP_FLOW=false
USE_DATA_DIR=false
FRESH=false
MULTIVIEW=false
PLAN_A=false
DATA_DIR=""
DATASET_NAME=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --data_dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --name)
            DATASET_NAME="$2"
            shift 2
            ;;
        --fresh)
            FRESH=true
            shift
            ;;
        --skip-vae)
            SKIP_VAE=true
            shift
            ;;
        --skip-flow)
            SKIP_FLOW=true
            shift
            ;;
        --use-data-dir)
            USE_DATA_DIR=true
            shift
            ;;
        --multiview)
            MULTIVIEW=true
            shift
            ;;
        --plan-a)
            PLAN_A=true
            shift
            ;;
        *)
            echo "未知参数: $1"
            echo ""
            echo "用法: bash scripts/run_finetune.sh [选项]"
            echo ""
            echo "选项:"
            echo "  --data_dir PATH    训练数据目录 (必需)"
            echo "  --name NAME        数据集名称 (可选，默认自动生成)"
            echo "  --fresh            从头开始训练 (忽略之前的微调结果)"
            echo "  --skip-vae         跳过 Shape VAE 微调"
            echo "  --skip-flow        跳过 Shape Flow 微调"
            echo "  --use-data-dir     输出到 /data/TRELLIS.2/"
            echo "  --multiview        多图模式 (每步随机 1~4 视角联合训练)"
            echo "  --plan-a           Flow 用预训练 latents/decoder 训练 (默认自动选 Plan B，并跳过 Step 4 重编码)"
            echo ""
            echo "数据支持:"
            echo "  Flat 模式 (--data_dir 下直接是 .glb/.obj)"
            echo "  子目录模式 (--data_dir 下是若干子目录，每个含恰好一个 .glb/.obj)"
            echo "  自动检测图片: 有则训练 VAE + Flow，无则仅训练 VAE"
            exit 1
            ;;
    esac
done

if [ -z "$DATA_DIR" ]; then
    echo "[错误] 必须指定 --data_dir"
    exit 1
fi

if [ -z "$DATASET_NAME" ]; then
    DATASET_NAME=$(echo "$DATA_DIR" | sed 's|/$||' | sed 's|^\./||' | tr '/' '_' | tr ' ' '_')
fi

# 数据目录指纹: 模型数量、首尾文件名及其 SHA256 头部。
compute_fingerprint() {
    DATA_DIR="$DATA_DIR" DATA_MODE="$DATA_MODE" python3 -c "
import os, hashlib
data_dir = os.environ['DATA_DIR']
data_mode = os.environ['DATA_MODE']
is_model = lambda f: f.lower().endswith(('.glb', '.obj'))
if data_mode == 'flat':
    models = sorted([f for f in os.listdir(data_dir) if is_model(f)])
else:
    subdirs = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    models = []
    for sd in subdirs:
        sp = os.path.join(data_dir, sd)
        candidates = sorted(f for f in os.listdir(sp) if is_model(f))
        if len(candidates) != 1:
            raise SystemExit(f'{sd}: expected exactly one .glb/.obj, found {len(candidates)}')
        models.append(os.path.join(sd, candidates[0]))
def sha256_head(path, n=65536):
    h = hashlib.sha256()
    with open(os.path.join(data_dir, path), 'rb') as fh: h.update(fh.read(n))
    return h.hexdigest()
if not models:
    raise SystemExit('no .glb/.obj source models found')
first = models[0]; last = models[-1]
print(f'{len(models)}|{first}|{last}|{sha256_head(first)}|{sha256_head(last)}')
"
}

# Flat 模式: 路径下直接有 .glb/.obj。
FLAT_MODEL_COUNT=$(find "$DATA_DIR" -maxdepth 1 -type f \( -iname "*.glb" -o -iname "*.obj" \) 2>/dev/null | wc -l)
SUBDIR_COUNT=$(find "$DATA_DIR" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)

if [ "$FLAT_MODEL_COUNT" -gt 0 ]; then
    DATA_MODE="flat"
    USE_FLAT="--flat"
    NUM_MODELS=$FLAT_MODEL_COUNT
    NUM_DIRS=0
else
    DATA_MODE="subdir"
    USE_FLAT=""
    NUM_MODELS=$(find "$DATA_DIR" -mindepth 2 -maxdepth 2 -type f \( -iname "*.glb" -o -iname "*.obj" \) 2>/dev/null | wc -l)
    NUM_DIRS=$SUBDIR_COUNT
fi

NUM_IMG=$(find "$DATA_DIR" -type f \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.webp" \) 2>/dev/null | wc -l)
HAS_IMAGES=false
if [ "$NUM_IMG" -gt 0 ]; then
    HAS_IMAGES=true
fi

# =============================================================================
# 环境设置
# =============================================================================
export CUDA_VISIBLE_DEVICES=7
export TMPDIR="${TMPDIR:-/data/tmp}"

mkdir -p "$TMPDIR" 2>/dev/null || {
    echo "[警告] 无法创建 TMPDIR=$TMPDIR，尝试使用 /tmp"
    export TMPDIR=/tmp
    mkdir -p "$TMPDIR" 2>/dev/null || true
}

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"
echo "工作目录: $(pwd)"

# =============================================================================
# 路径配置
# =============================================================================
OUTPUT_DIR="./training_data/$DATASET_NAME"
VAE_OUTPUT="./results/${DATASET_NAME}_vae"
FLOW_OUTPUT="./results/${DATASET_NAME}_flow"
CKPT_DIR="checkpoints/ckpts"

SHAPE_ENC_PT="$CKPT_DIR/shape_enc_next_dc_f16c32_fp16.pt"
SHAPE_DEC_PT="$CKPT_DIR/shape_dec_next_dc_f16c32_fp16.pt"
SLAT_FLOW_PT="$CKPT_DIR/slat_flow_img2shape_dit_1_3B_512_bf16.pt"

LATEST_VAE_DIR="./results/latest_vae"
LATEST_ENC="$LATEST_VAE_DIR/ckpts/encoder_latest.pt"
LATEST_DEC="$LATEST_VAE_DIR/ckpts/decoder_latest.pt"
LATEST_FLOW_DIR="./results/latest_flow"
LATEST_FLOW_CKPT="$LATEST_FLOW_DIR/ckpts/denoiser_latest.pt"

if $FRESH; then
    FT_ENC="$SHAPE_ENC_PT"
    FT_DEC="$SHAPE_DEC_PT"
    VAE_START_MODE="从头开始 (预训练权重)"
elif [ -f "$LATEST_ENC" ] && [ -f "$LATEST_DEC" ]; then
    FT_ENC="$LATEST_ENC"
    FT_DEC="$LATEST_DEC"
    VAE_START_MODE="接力训练 (latest_vae)"
else
    FT_ENC="$SHAPE_ENC_PT"
    FT_DEC="$SHAPE_DEC_PT"
    VAE_START_MODE="首次训练 (预训练权重)"
fi

if $FRESH; then
    FT_FLOW="$SLAT_FLOW_PT"
    FLOW_START_MODE="从头开始 (预训练权重)"
elif [ -f "$LATEST_FLOW_CKPT" ]; then
    FT_FLOW="$LATEST_FLOW_CKPT"
    FLOW_START_MODE="接力训练 (latest_flow)"
else
    FT_FLOW="$SLAT_FLOW_PT"
    FLOW_START_MODE="首次训练 (预训练权重)"
fi

if $USE_DATA_DIR; then
    DATA_BASE="/data/TRELLIS.2"
    mkdir -p "$DATA_BASE/results" "$DATA_BASE/training_data"
    for d in results training_data; do
        if [ ! -L "./$d" ] && [ -d "./$d" ]; then
            mv "./$d"/* "$DATA_BASE/$d/" 2>/dev/null || true
            rm -rf "./$d"
        fi
        if [ ! -e "./$d" ]; then
            ln -s "$DATA_BASE/$d" "./$d"
        fi
    done
fi

# =============================================================================
# =============================================================================
echo ""
echo "============================================================================"
echo "  TRELLIS.2 迭代微调"
echo "============================================================================"
echo ""
echo "数据集:      $DATASET_NAME"
echo "数据格式:    $DATA_MODE 模式 ($NUM_MODELS 个 GLB/OBJ, $NUM_IMG 张图片)"
if $SKIP_VAE; then VAE_STATUS='(跳过)'; else VAE_STATUS='(进行)'; fi
if $SKIP_FLOW; then FLOW_STATUS='(跳过)'; elif $HAS_IMAGES; then FLOW_STATUS='(进行)'; else FLOW_STATUS='(无图片,跳过)'; fi
echo "训练阶段:    VAE $VAE_STATUS | Flow $FLOW_STATUS"
echo ""
echo "VAE 起点:    $VAE_START_MODE"
echo "  encoder:   $FT_ENC"
echo "  decoder:   $FT_DEC"
if $HAS_IMAGES && ! $SKIP_FLOW; then
    echo "Flow 起点:   $FLOW_START_MODE"
    echo "  denoiser:  $FT_FLOW"
fi
echo ""
echo "路径:"
echo "  原始数据:   $DATA_DIR"
echo "  预处理:     $OUTPUT_DIR"
echo "  VAE 输出:   $VAE_OUTPUT"
echo "  Flow 输出:  $FLOW_OUTPUT"
echo ""

# =============================================================================
# 预检查
# =============================================================================
echo "============================================================================"
echo "预检查"
echo "============================================================================"

if ! python -c "import torch, o_voxel, trimesh" 2>/dev/null; then
    echo "[错误] 请先激活 conda 环境: conda activate trellis2"
    exit 1
fi
echo "  ✓ conda 环境"

if [ ! -d "$DATA_DIR" ]; then
    echo "[错误] 数据目录不存在: $DATA_DIR"
    exit 1
fi
if [ "$DATA_MODE" = "flat" ]; then
    echo "  ✓ 模型文件: $NUM_MODELS 个 (Flat 模式)"
else
    echo "  ✓ 子目录: $NUM_DIRS 个, 模型文件: $NUM_MODELS 个"
    python3 scripts/check_data_dirs.py --data_dir "$DATA_DIR" --expect_views 24
fi
echo "  ✓ 图片文件: $NUM_IMG 张"
if [ "$NUM_MODELS" -eq 0 ]; then
    echo "[错误] 没有任何 .glb/.obj 文件!"
    exit 1
fi

for f in "$SHAPE_ENC_PT" "$SHAPE_DEC_PT"; do
    if [ ! -f "$f" ]; then
        echo "[错误] 预训练权重不存在: $f"
        exit 1
    fi
done
echo "  ✓ VAE 预训练权重"

if $HAS_IMAGES && ! $SKIP_FLOW; then
    if [ ! -f "$SLAT_FLOW_PT" ]; then
        echo "[错误] Flow 预训练权重不存在: $SLAT_FLOW_PT"
        exit 1
    fi
    if [ ! -f "./dinov3/model.safetensors" ] && [ ! -f "./dinov3/pytorch_model.bin" ]; then
        echo "[错误] DINOv3 本地模型不存在: ./dinov3/"
        exit 1
    fi
    echo "  ✓ Flow 预训练权重 + DINOv3"
fi

ROOT_AVAIL=$(df -BG . 2>/dev/null | tail -1 | awk '{print $4}' | sed 's/G//')
echo "  ✓ 磁盘可用: ${ROOT_AVAIL}G"
if [ "$ROOT_AVAIL" -lt 20 ] && ! $USE_DATA_DIR; then
    echo "  [警告] 空间不足 20G，建议加 --use-data-dir"
fi

echo ""
echo "============================================================================"
echo "应用补丁 & 格式检查"
echo "============================================================================"
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
python3 scripts/apply_patches.py

NEED_CONVERT=false
for f in "$SHAPE_ENC_PT" "$SHAPE_DEC_PT" "$SLAT_FLOW_PT"; do
    if [ ! -f "$f" ]; then NEED_CONVERT=true; break; fi
done
if $NEED_CONVERT; then
    echo "  转换 safetensors → .pt ..."
    python scripts/convert_safetensors_to_pt.py
fi
echo "  ✓ 完成"

# =============================================================================
# Step 1: 数据预处理
# =============================================================================
echo ""
echo "============================================================================"
echo "Step 1: 数据预处理 (GLB/OBJ -> dual_grid + mesh_dump)"
echo "============================================================================"

MARKER_FILE="$OUTPUT_DIR/.preprocess_marker"
NEED_PREPARE=false
for f in "dual_grid_512" "mesh_dumps" "metadata.csv"; do
    if [ ! -e "$OUTPUT_DIR/$f" ]; then
        NEED_PREPARE=true; break
    fi
done

# 指纹过期检测
if [ -f "$OUTPUT_DIR/metadata.csv" ] && [ ! -f "$MARKER_FILE" ]; then
    echo "  [警告] 预处理数据缺少指纹标记，将重新预处理"
    rm -rf "$OUTPUT_DIR/dual_grid_512" "$OUTPUT_DIR/mesh_dumps" "$OUTPUT_DIR/metadata.csv" \
           "$OUTPUT_DIR/shape_latents" "$OUTPUT_DIR/renders_cond"
    NEED_PREPARE=true
elif [ -f "$MARKER_FILE" ] && [ -f "$OUTPUT_DIR/metadata.csv" ]; then
    CURRENT_FP=$(compute_fingerprint 2>/dev/null)
    STORED_FP=$(cat "$MARKER_FILE" 2>/dev/null)
    if [ "$CURRENT_FP" != "$STORED_FP" ] && [ -n "$CURRENT_FP" ]; then
        echo "  [警告] 数据已变更，将重新预处理"
        rm -rf "$OUTPUT_DIR/dual_grid_512" "$OUTPUT_DIR/mesh_dumps" "$OUTPUT_DIR/metadata.csv" \
               "$OUTPUT_DIR/shape_latents" "$OUTPUT_DIR/renders_cond" "$MARKER_FILE"
        NEED_PREPARE=true
    fi
fi

if $NEED_PREPARE; then
    python data_toolkit/custom_prepare_data.py \
        --data_dir "$DATA_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --resolutions 512 \
        $USE_FLAT
    echo "[完成] 数据预处理"

    FP=$(compute_fingerprint 2>/dev/null)
    echo "$FP" > "$MARKER_FILE"
else
    echo "[跳过] 预处理结果已存在"
fi

if [ -f "$OUTPUT_DIR/metadata.csv" ]; then
    for sub in dual_grid_512 mesh_dumps asset_stats renders_cond; do
        mkdir -p "$OUTPUT_DIR/$sub"
        cp "$OUTPUT_DIR/metadata.csv" "$OUTPUT_DIR/$sub/"
    done
fi

# 修复 renders_cond/transforms.json
if $HAS_IMAGES; then
    RENDERS_DIR="$OUTPUT_DIR/renders_cond"
    FIRST_TRANSFORM=$(find "$RENDERS_DIR" -name "transforms.json" 2>/dev/null | head -1)
    if [ -n "$FIRST_TRANSFORM" ]; then
        FILE_PATH=$(python3 -c "import json; d=json.load(open('$FIRST_TRANSFORM')); print(d['frames'][0]['file_path'])")
        if [ "$FILE_PATH" = "000" ]; then
            find "$RENDERS_DIR" -name "transforms.json" -exec python3 -c "
import json, sys
for fp in sys.argv[1:]:
    with open(fp) as f: d = json.load(f)
    for frame in d['frames']: frame['file_path'] += '.png' if not frame['file_path'].endswith('.png') else ''
    with open(fp, 'w') as f: json.dump(d, f, indent=2)
" {} +
        fi
    fi
fi

# =============================================================================
# Step 2: 编码 shape latents
# =============================================================================
echo ""
echo "============================================================================"
echo "Step 2: 编码 shape latents"
echo "============================================================================"

LATENT_DIR="$OUTPUT_DIR/shape_latents/shape_enc_next_dc_f16c32_fp16_512"
NPZ_COUNT=$(find "$LATENT_DIR" -name "*.npz" 2>/dev/null | wc -l)

if [ ! -f "$LATENT_DIR/check_done.txt" ] || [ "$NPZ_COUNT" -eq 0 ]; then
    if [ "$NPZ_COUNT" -eq 0 ] && [ -f "$LATENT_DIR/check_done.txt" ]; then
        echo "  [警告] check_done.txt 存在但无 .npz 文件，重新编码"
    fi
    rm -f "$LATENT_DIR/check_done.txt"
    python data_toolkit/encode_shape_latent.py \
        --root "$OUTPUT_DIR" \
        --resolution 512 \
        --enc_pretrained "$CKPT_DIR/shape_enc_next_dc_f16c32_fp16"

    python3 -c "
import pandas as pd, os
meta_path = '$OUTPUT_DIR/metadata.csv'
r = '$LATENT_DIR/new_records/part_0.csv'
if os.path.exists(r) and os.path.getsize(r) > 0:
    try:
        m = pd.read_csv(meta_path).set_index('sha256')
        n = pd.read_csv(r).set_index('sha256')
        if len(n) > 0:
            m = n.combine_first(m).reset_index()
            m.to_csv(meta_path, index=False)
            print(f'metadata 合并: {len(m)} 条')
    except Exception as e:
        print(f'[警告] {e}')
"
    touch "$LATENT_DIR/check_done.txt"
    echo "[完成] shape latents 编码"
else
    echo "[跳过] shape latents 已存在"
fi

# 确保 metadata.csv 在 latent 目录中 (Flow 训练读取)
if [ -f "$OUTPUT_DIR/metadata.csv" ]; then
    cp "$OUTPUT_DIR/metadata.csv" "$LATENT_DIR/"
fi

# =============================================================================
# Step 3: 微调 Shape VAE
# =============================================================================
echo ""
echo "============================================================================"
VAE_MAX_STEPS=$(python3 -c "import json; cfg=json.load(open('configs/scvae/shape_vae_ft_train.json')); print(cfg['trainer']['args']['max_steps'])")
echo "Step 3: 微调 Shape VAE (约 1-2 小时, max_steps=$VAE_MAX_STEPS)"
echo "============================================================================"

if $SKIP_VAE; then
    echo "[跳过] 用户指定 --skip-vae"
else
    echo "起点: $VAE_START_MODE"

    # 计算最终 checkpoint step
    VAE_SAVE_INTERVAL=$(python3 -c "import json; cfg=json.load(open('configs/scvae/shape_vae_ft_train.json')); print(cfg['trainer']['args']['i_save'])")
    VAE_FINAL_STEP=$(python3 -c "ms=$VAE_MAX_STEPS; iv=$VAE_SAVE_INTERVAL; print((ms // iv) * iv)")
    VAE_FINAL_MISC="$VAE_OUTPUT/ckpts/misc_step$(printf '%07d' $VAE_FINAL_STEP).pt"

    VAE_FINGERPRINT="$VAE_OUTPUT/.data_fingerprint"
    NEED_VAE_TRAIN=true

    if $FRESH; then
        echo "  [--fresh] 强制从头训练 VAE"
        rm -rf "$VAE_OUTPUT/ckpts" "$VAE_FINGERPRINT"
    elif [ -f "$VAE_FINAL_MISC" ]; then
        if [ -f "$VAE_FINGERPRINT" ] && [ -f "$MARKER_FILE" ]; then
            if [ "$(cat "$VAE_FINGERPRINT")" = "$(cat "$MARKER_FILE")" ]; then
                echo "[跳过] Shape VAE 已训练完成 (max_steps=$VAE_MAX_STEPS) 且数据指纹一致"
                NEED_VAE_TRAIN=false
            else
                echo "  [警告] VAE checkpoint 对应旧数据，将重新训练"
                rm -rf "$VAE_OUTPUT/ckpts" "$VAE_FINGERPRINT"
            fi
        else
            echo "  [警告] VAE checkpoint 缺少指纹，将重新训练"
            rm -rf "$VAE_OUTPUT/ckpts"
        fi
    fi

    if $NEED_VAE_TRAIN; then
        TEMP_CONFIG="/tmp/shape_vae_ft_${DATASET_NAME}.json"
        FT_ENC="$FT_ENC" FT_DEC="$FT_DEC" TEMP_CONFIG="$TEMP_CONFIG" python3 -c "
import json, os
with open('configs/scvae/shape_vae_ft_train.json') as f:
    cfg = json.load(f)
cfg['trainer']['args']['finetune_ckpt'] = {
    'encoder': os.environ['FT_ENC'],
    'decoder': os.environ['FT_DEC']
}
with open(os.environ['TEMP_CONFIG'], 'w') as f:
    json.dump(cfg, f, indent=2)
"

        python train.py \
            --config "$TEMP_CONFIG" \
            --output_dir "$VAE_OUTPUT" \
            --data_dir "{\"train_data\":{\"base\":\"$OUTPUT_DIR\",\"mesh_dump\":\"$OUTPUT_DIR/mesh_dumps\",\"dual_grid\":\"$OUTPUT_DIR/dual_grid_512\",\"asset_stats\":\"$OUTPUT_DIR/asset_stats\"}}"

        echo "[完成] Shape VAE 微调"
        rm -f "$TEMP_CONFIG"

        if [ -f "$MARKER_FILE" ]; then
            cp "$MARKER_FILE" "$VAE_FINGERPRINT"
        fi

        mkdir -p "$LATEST_VAE_DIR/ckpts"
        LATEST_ENC_CKPT=$(ls "$VAE_OUTPUT/ckpts/encoder_step"*.pt 2>/dev/null | sort -V | tail -1)
        # 使用非 EMA 的 decoder 权重: ema_rate=0.9999 时 EMA 权重仍含 ~82% 预训练
        # (0.9999^2000 ≈ 0.82)，非 EMA 权重才是完整微调结果
        LATEST_DEC_CKPT=$(ls "$VAE_OUTPUT/ckpts/decoder_step"*.pt 2>/dev/null | sort -V | tail -1)
        if [ -z "$LATEST_DEC_CKPT" ]; then
            LATEST_DEC_CKPT=$(ls "$VAE_OUTPUT/ckpts/decoder_ema"*.pt 2>/dev/null | sort -V | tail -1)
        fi
        if [ -n "$LATEST_ENC_CKPT" ] && [ -n "$LATEST_DEC_CKPT" ]; then
            cp "$LATEST_ENC_CKPT" "$LATEST_ENC"
            cp "$LATEST_DEC_CKPT" "$LATEST_DEC"
            echo "  ✓ 已更新 latest_vae → $LATEST_VAE_DIR/ (decoder: $(basename "$LATEST_DEC_CKPT"))"
        fi
    fi

    VAE_ENC_CKPT=$(ls "$VAE_OUTPUT/ckpts/encoder_step"*.pt 2>/dev/null | sort -V | tail -1)
    if [ -z "$VAE_ENC_CKPT" ]; then
        echo "[错误] 找不到 VAE encoder checkpoint!"
        exit 1
    fi
    VAE_STEP=$(basename "$VAE_ENC_CKPT" | sed 's/encoder_//' | sed 's/\.pt//')
    echo "VAE checkpoint step: $VAE_STEP"
fi

# =============================================================================
# Step 4: 用微调后 encoder 重新编码 latents (可选, 仅当 VAE 训练了)
# =============================================================================
if $SKIP_VAE || $PLAN_A; then
    echo ""
    if $PLAN_A; then
        echo "Step 4: 跳过 (--plan-a: Flow 用预训练 latents，无需微调后 encoder 重编码)"
    else
        echo "Step 4: 跳过 (VAE 未训练)"
    fi
    FT_LATENT_DIR=""
else
echo ""
echo "============================================================================"
echo "Step 4: 重新编码 shape latents (微调后 encoder)"
echo "============================================================================"

VAE_PARENT=$(dirname "$VAE_OUTPUT")
VAE_DIRNAME=$(basename "$VAE_OUTPUT")
FT_LATENT_DIRNAME="${VAE_DIRNAME}_${VAE_STEP}_512"
FT_LATENT_DIR="$OUTPUT_DIR/shape_latents/$FT_LATENT_DIRNAME"
FT_NPZ_COUNT=$(find "$FT_LATENT_DIR" -name "*.npz" 2>/dev/null | wc -l)

if [ ! -f "$FT_LATENT_DIR/check_done.txt" ] || [ "$FT_NPZ_COUNT" -eq 0 ]; then
    if [ "$FT_NPZ_COUNT" -eq 0 ] && [ -f "$FT_LATENT_DIR/check_done.txt" ]; then
        echo "  [警告] check_done.txt 存在但无 .npz 文件，重新编码"
    fi
    rm -f "$FT_LATENT_DIR/check_done.txt"
    # 清理上次失败残留的 metadata.csv，否则 encode_shape_latent.py
    # 会通过 combine_first 把 shape_latent_encoded=True 还原回来
    rm -f "$FT_LATENT_DIR/metadata.csv"
    # Step 3 编码后在 metadata 中写入了 shape_latent_encoded=True，
    # 会导致 encode_shape_latent.py 过滤掉所有样本。临时去掉该列。
    python3 -c "
import pandas as pd
m = pd.read_csv('$OUTPUT_DIR/metadata.csv')
m.drop(columns=['shape_latent_encoded'], inplace=True, errors='ignore')
m.to_csv('$OUTPUT_DIR/metadata.csv', index=False)
"
    python data_toolkit/encode_shape_latent.py \
        --root "$OUTPUT_DIR" \
        --resolution 512 \
        --model_root "$VAE_PARENT" \
        --enc_model "$VAE_DIRNAME" \
        --ckpt "$VAE_STEP"

    FT_ACTUAL_NPZ=$(find "$FT_LATENT_DIR" -name "*.npz" 2>/dev/null | wc -l)
    if [ "$FT_ACTUAL_NPZ" -eq 0 ]; then
        echo "  [错误] 编码命令执行后仍未生成 .npz 文件!"
        echo "  目录: $FT_LATENT_DIR"
        echo "  请检查 encode_shape_latent.py 的错误输出"
        # 不创建 check_done.txt，下次运行会重试
    else
        echo "  [完成] 编码完成 ($FT_ACTUAL_NPZ 个 .npz 文件)"
        touch "$FT_LATENT_DIR/check_done.txt"
    fi

    python3 -c "
import pandas as pd, os
meta_path = '$OUTPUT_DIR/metadata.csv'
r = os.path.join('$FT_LATENT_DIR', 'new_records', 'part_0.csv')
if os.path.exists(r) and os.path.getsize(r) > 0:
    try:
        m = pd.read_csv(meta_path).set_index('sha256')
        n = pd.read_csv(r).set_index('sha256')
        if len(n) > 0:
            m = n.combine_first(m).reset_index()
            m.to_csv(meta_path, index=False)
            print(f'metadata 合并: {len(m)} 条')
    except Exception as e:
        print(f'[警告] {e}')
"
else
    echo "[跳过] 微调后 latents 已存在"
fi

# 确保 metadata.csv 在 latent 目录中
if [ -f "$OUTPUT_DIR/metadata.csv" ] && [ -n "$FT_LATENT_DIR" ]; then
    cp "$OUTPUT_DIR/metadata.csv" "$FT_LATENT_DIR/"
fi
fi  # end of if ! ($SKIP_VAE || $PLAN_A) (Step 4)

# =============================================================================
# Step 5: 微调 Shape Flow
# =============================================================================
echo ""
echo "============================================================================"
echo "Step 5: 微调 Shape Flow (约 2-4 小时, 步数由配置文件决定)"
echo "============================================================================"

if $SKIP_FLOW; then
    echo "[跳过] 用户指定 --skip-flow"
elif ! $HAS_IMAGES; then
    echo "[跳过] 无图片，无法训练 Shape Flow"
elif [ ! -d "$OUTPUT_DIR/renders_cond" ] || [ -z "$(ls -A "$OUTPUT_DIR/renders_cond" 2>/dev/null)" ]; then
    echo "[跳过] renders_cond 为空，无法训练 Shape Flow"
else
    echo "起点: $FLOW_START_MODE"

    # Plan A / Plan B 选择 (多图模式使用对应的 multiview 配置)
    FT_FLOW_NPZ=$(find "$FT_LATENT_DIR" -name "*.npz" 2>/dev/null | wc -l)
    if $MULTIVIEW; then
        if ! $PLAN_A && ! $SKIP_VAE && [ -d "$FT_LATENT_DIR" ] && [ "$FT_FLOW_NPZ" -gt 0 ]; then
            FLOW_CONFIG="configs/gen/shape_flow_ft_vae_multiview.json"
            FLOW_LATENT_DIR="$FT_LATENT_DIR"
            FLOW_PLAN="B-多图 (微调后 VAE decoder + 微调后 latents + 多视角训练)"
        else
            if [ -d "$FT_LATENT_DIR" ] && [ "$FT_FLOW_NPZ" -eq 0 ]; then
                echo "  [注意] FT_LATENT_DIR 存在但无 .npz 文件，使用方案 A-多图"
            fi
            FLOW_CONFIG="configs/gen/shape_flow_ft_multiview.json"
            FLOW_LATENT_DIR="$OUTPUT_DIR/shape_latents/shape_enc_next_dc_f16c32_fp16_512"
            FLOW_PLAN="A-多图 (原始 VAE decoder + 原始 latents + 多视角训练)"
        fi
    else
        if ! $PLAN_A && ! $SKIP_VAE && [ -d "$FT_LATENT_DIR" ] && [ "$FT_FLOW_NPZ" -gt 0 ]; then
            FLOW_CONFIG="configs/gen/shape_flow_ft_vae_stage2.json"
            FLOW_LATENT_DIR="$FT_LATENT_DIR"
            FLOW_PLAN="B (微调后 VAE decoder + 微调后 latents, $FT_FLOW_NPZ 个 npz)"
        else
            if [ -d "$FT_LATENT_DIR" ] && [ "$FT_FLOW_NPZ" -eq 0 ]; then
                echo "  [注意] FT_LATENT_DIR 存在但无 .npz 文件，使用方案 A"
            fi
            FLOW_CONFIG="configs/gen/shape_flow_ft_train.json"
            FLOW_LATENT_DIR="$OUTPUT_DIR/shape_latents/shape_enc_next_dc_f16c32_fp16_512"
            FLOW_PLAN="A (原始 VAE decoder + 原始 latents)"
        fi
    fi

    # 读取 max_steps 确定最终 checkpoint
    FLOW_MAX_STEPS=$(python3 -c "import json; cfg=json.load(open('$FLOW_CONFIG')); print(cfg['trainer']['args']['max_steps'])")
    FLOW_SAVE_INTERVAL=$(python3 -c "import json; cfg=json.load(open('$FLOW_CONFIG')); print(cfg['trainer']['args']['i_save'])")
    FLOW_FINAL_STEP=$(python3 -c "ms=$FLOW_MAX_STEPS; iv=$FLOW_SAVE_INTERVAL; print((ms // iv) * iv)")
    FLOW_FINAL_MISC="$FLOW_OUTPUT/ckpts/misc_step$(printf '%07d' $FLOW_FINAL_STEP).pt"

    FLOW_FINGERPRINT="$FLOW_OUTPUT/.data_fingerprint"
    NEED_FLOW_TRAIN=true

    if $FRESH; then
        echo "  [--fresh] 强制从头训练 Flow"
        rm -rf "$FLOW_OUTPUT/ckpts" "$FLOW_FINGERPRINT"
    elif [ -f "$FLOW_FINAL_MISC" ]; then
        if [ -f "$FLOW_FINGERPRINT" ] && [ -f "$MARKER_FILE" ]; then
            if [ "$(cat "$FLOW_FINGERPRINT")" = "$(cat "$MARKER_FILE")" ]; then
                echo "[跳过] Shape Flow 已训练完成 (max_steps=$FLOW_MAX_STEPS) 且数据指纹一致"
                NEED_FLOW_TRAIN=false
            else
                echo "  [警告] Flow checkpoint 对应旧数据，将重新训练"
                rm -rf "$FLOW_OUTPUT/ckpts" "$FLOW_FINGERPRINT"
            fi
        else
            echo "  [警告] Flow checkpoint 缺少指纹，将重新训练"
            rm -rf "$FLOW_OUTPUT/ckpts"
        fi
    fi

    if $NEED_FLOW_TRAIN; then
        COND_COUNT=$(find "$OUTPUT_DIR/renders_cond" -name "transforms.json" 2>/dev/null | wc -l)
        echo "  方案: $FLOW_PLAN | 样本: $COND_COUNT | steps: $FLOW_MAX_STEPS"
        echo "  Config: $FLOW_CONFIG"
        echo ""

        TEMP_FLOW_CONFIG="/tmp/shape_flow_ft_${DATASET_NAME}.json"
        FT_FLOW="$FT_FLOW" FLOW_CONFIG="$FLOW_CONFIG" \
          VAE_OUTPUT="$VAE_OUTPUT" VAE_STEP="$VAE_STEP" LATEST_DEC="$LATEST_DEC" \
          TEMP_FLOW_CONFIG="$TEMP_FLOW_CONFIG" python3 -c "
import json, os
with open(os.environ['FLOW_CONFIG']) as f:
    cfg = json.load(f)
cfg['trainer']['args']['finetune_ckpt'] = {'denoiser': os.environ['FT_FLOW']}
# Plan B: 配置中有 slat_dec_path 字段 → 需要更新为实际 VAE 输出路径
if 'slat_dec_path' in cfg.get('dataset', {}).get('args', {}):
    cfg['dataset']['args']['slat_dec_path'] = os.environ['VAE_OUTPUT']
    cfg['dataset']['args']['slat_dec_ckpt'] = os.environ['VAE_STEP']
with open(os.environ['TEMP_FLOW_CONFIG'], 'w') as f:
    json.dump(cfg, f, indent=2)
"

        python train.py \
            --config "$TEMP_FLOW_CONFIG" \
            --output_dir "$FLOW_OUTPUT" \
            --data_dir "{\"train_data\":{\"base\":\"$OUTPUT_DIR\",\"shape_latent\":\"$FLOW_LATENT_DIR\",\"render_cond\":\"$OUTPUT_DIR/renders_cond\"}}"

        rm -f "$TEMP_FLOW_CONFIG"

        echo "[完成] Shape Flow 微调"

        if [ -f "$MARKER_FILE" ]; then
            cp "$MARKER_FILE" "$FLOW_FINGERPRINT"
        fi

        mkdir -p "$LATEST_FLOW_DIR/ckpts"
        # 从配置读取 ema_rate (当前为 0.999)，与保存的 checkpoint 文件名一致
        FLOW_EMA_RATE=$(python3 -c "import json; cfg=json.load(open('$FLOW_CONFIG')); r=cfg['trainer']['args'].get('ema_rate',[0.999]); print(r[0] if isinstance(r,list) else r)")
        LATEST_FLOW_NEW=$(ls "$FLOW_OUTPUT/ckpts/denoiser_ema${FLOW_EMA_RATE}_step"*.pt 2>/dev/null | sort -V | tail -1)
        if [ -z "$LATEST_FLOW_NEW" ]; then
            LATEST_FLOW_NEW=$(ls "$FLOW_OUTPUT/ckpts/denoiser_step"*.pt 2>/dev/null | sort -V | tail -1)
        fi
        if [ -n "$LATEST_FLOW_NEW" ]; then
            cp "$LATEST_FLOW_NEW" "$LATEST_FLOW_CKPT"
            echo "  ✓ 已更新 latest_flow → $LATEST_FLOW_DIR/ ($(basename "$LATEST_FLOW_NEW"))"
        fi
    fi
fi

# =============================================================================
# 完成
# =============================================================================
echo ""
echo "============================================================================"
echo "  ✓ 微调完成!"
echo "============================================================================"
echo ""
echo "数据集:      $DATASET_NAME"
echo "数据格式:    $DATA_MODE ($NUM_MODELS 个 GLB/OBJ, $NUM_IMG 张图片)"
echo ""
if ! $SKIP_VAE; then
    echo "Shape VAE:"
    echo "  本次输出:   $VAE_OUTPUT/ckpts/encoder_${VAE_STEP}.pt"
    echo "  Latest:     $LATEST_ENC"
    echo ""
fi
if $HAS_IMAGES && ! $SKIP_FLOW && [ -f "$FLOW_FINAL_MISC" ]; then
    echo "Shape Flow:"
    echo "  本次输出:   $FLOW_OUTPUT/ckpts/denoiser_step$(printf '%07d' $FLOW_FINAL_STEP).pt"
    echo "  Latest:     $LATEST_FLOW_CKPT"
    echo ""
fi
echo "下次训练将自动从 latest 权重继续 (--fresh 从头开始)"

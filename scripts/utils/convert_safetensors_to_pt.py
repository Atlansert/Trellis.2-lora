#!/usr/bin/env python3
"""
将 HuggingFace safetensors 格式的预训练模型转换为 .pt 格式
用于 finetune_ckpt 加载

服务器 C 上的预训练权重位于 checkpoints/ckpts/ 目录

用法 (服务器 C 上):
    cd /home/lyx/lzx/TRELLIS.2
    python scripts/convert_safetensors_to_pt.py

输出:
    checkpoints/ckpts/shape_enc_next_dc_f16c32_fp16.pt
    checkpoints/ckpts/shape_dec_next_dc_f16c32_fp16.pt
    checkpoints/ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.pt
    checkpoints/ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16.pt
"""

import os
import sys
import torch

# 限制只使用 GPU 7
os.environ['CUDA_VISIBLE_DEVICES'] = '7'

# 添加项目根目录到 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trellis2 import models


def convert_safetensors_to_pt(local_path: str, output_path: str):
    """
    从本地 safetensors 加载模型，保存为 .pt 格式

    Args:
        local_path: 本地路径前缀，如 "checkpoints/ckpts/shape_enc_next_dc_f16c32_fp16"
        output_path: 输出的 .pt 文件路径
    """
    print(f"\n{'='*60}")
    print(f"加载: {local_path}")

    # 检查本地文件
    if not (os.path.exists(f"{local_path}.json") and os.path.exists(f"{local_path}.safetensors")):
        raise FileNotFoundError(
            f"找不到本地模型文件:\n"
            f"  {local_path}.json\n"
            f"  {local_path}.safetensors\n"
            f"请确认 checkpoints/ckpts/ 目录下有这些文件"
        )

    print(f"  → 从本地加载")
    model = models.from_pretrained(local_path)
    state_dict = model.state_dict()

    # 保存为 .pt
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(state_dict, output_path)

    # 统计
    num_params = sum(v.numel() for v in state_dict.values())
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  → 保存: {output_path}")
    print(f"  → 参数量: {num_params:,} ({file_size_mb:.1f} MB)")

    del model, state_dict
    torch.cuda.empty_cache()


def main():
    # 转换列表: (本地路径前缀, 输出路径)
    # 服务器 C 上 safetensors 在 checkpoints/ckpts/ 下
    conversions = [
        (
            "checkpoints/ckpts/shape_enc_next_dc_f16c32_fp16",
            "checkpoints/ckpts/shape_enc_next_dc_f16c32_fp16.pt",
        ),
        (
            "checkpoints/ckpts/shape_dec_next_dc_f16c32_fp16",
            "checkpoints/ckpts/shape_dec_next_dc_f16c32_fp16.pt",
        ),
        (
            "checkpoints/ckpts/slat_flow_img2shape_dit_1_3B_512_bf16",
            "checkpoints/ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.pt",
        ),
        (
            "checkpoints/ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16",
            "checkpoints/ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16.pt",
        ),
    ]

    print("=" * 60)
    print("TRELLIS.2 模型格式转换: safetensors → .pt")
    print("=" * 60)

    # 检查源文件是否存在
    for local_path, _ in conversions:
        json_file = f"{local_path}.json"
        safetensors_file = f"{local_path}.safetensors"
        if os.path.exists(json_file) and os.path.exists(safetensors_file):
            print(f"  ✓ 找到: {local_path}")
        else:
            print(f"  ✗ 缺失: {json_file if not os.path.exists(json_file) else safetensors_file}")
            print(f"\n[错误] 请确认 checkpoints/ckpts/ 目录下的文件是否完整")
            sys.exit(1)

    print(f"\n开始转换...")

    for local_path, output_name in conversions:
        if os.path.exists(output_name):
            size_mb = os.path.getsize(output_name) / (1024 * 1024)
            print(f"\n[跳过] {output_name} 已存在 ({size_mb:.1f} MB)")
            continue

        try:
            convert_safetensors_to_pt(local_path, output_name)
        except Exception as e:
            print(f"\n[错误] 转换失败: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    print(f"\n{'='*60}")
    print("✓ 所有转换完成!")
    print(f"{'='*60}")
    print("\n生成的文件:")
    for _, output_name in conversions:
        if os.path.exists(output_name):
            size_mb = os.path.getsize(output_name) / (1024 * 1024)
            print(f"  ✓ {output_name} ({size_mb:.1f} MB)")

    print("\n现在可以运行微调脚本: bash scripts/run_finetune.sh")


if __name__ == "__main__":
    main()

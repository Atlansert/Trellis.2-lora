#!/usr/bin/env python3
"""
使用 latest 训练权重进行 TRELLIS.2 推理
==========================================
自动检测 results/latest_vae/ 和 results/latest_flow/ 下的微调权重，
替换 pretrained pipeline 中的对应模型进行推理。

用法 (服务器 C):
    # 单张图片
    python scripts/infer_finetuned.py --image input.png --output output/

    # 批量处理目录
    python scripts/infer_finetuned.py --input_dir ./my_images --output ./results_infer/

    # 指定 GPU
    CUDA_VISIBLE_DEVICES=7 python scripts/infer_finetuned.py --image input.png --output output/

    # 只使用 VAE decoder，不替换 flow (仅 VAE 微调场景)
    python scripts/infer_finetuned.py --image input.png --output output/ --vae_only

    # 纯官方基线，不自动加载 results/latest_* 下的微调权重
    python scripts/infer_finetuned.py --image input.png --output output/ --pretrained_only
"""

import os
# Respect an explicitly selected GPU (e.g. run inference on GPU0 while
# training continues on GPU7); retain GPU7 as the standalone default.
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '7')
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import glob
import json

import torch
import imageio
import numpy as np
import trimesh
from PIL import Image

from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.utils import render_utils
from trellis2.renderers import EnvMap
from trellis2.models.sc_vaes.fdg_vae import FlexiDualGridVaeDecoder
from trellis2.modules.lora import apply_lora_to_linear_layers
import o_voxel


def load_finetuned_weights(pipeline, vae_decoder_path=None, flow_512_path=None,
                           flow_512_lora_path=None, flow_1024_path=None, flow_1024_lora_path=None,
                           tex_slat_lora_path=None,
                           sparse_structure_lora_path=None,
                           lora_rank=8, lora_alpha=8.0, lora_dropout=0.0,
                           ss_lora_rank=None, ss_lora_alpha=None,
                           shape_lora_rank=None, shape_lora_alpha=None,
                           tex_lora_rank=None, tex_lora_alpha=None):
    """
    将微调后的权重加载到 pipeline 中，替换对应的预训练模型。

    Args:
        pipeline: 已加载的 Trellis2ImageTo3DPipeline
        vae_decoder_path: VAE decoder checkpoint 路径 (.pt)
        flow_512_path: Shape Flow 512 checkpoint 路径 (.pt)
        flow_1024_path: Shape Flow 1024 checkpoint 路径 (.pt)
    """
    loaded = []

    # --- Sparse Structure Flow LoRA (TRELLIS.1-style Stage1 adaptation) ---
    if sparse_structure_lora_path and os.path.exists(sparse_structure_lora_path):
        print(f"  加载 Sparse Structure Flow LoRA: {sparse_structure_lora_path}")
        model = pipeline.models['sparse_structure_flow_model']
        replaced = apply_lora_to_linear_layers(
            model,
            ['self_attn.to_qkv', 'self_attn.to_out', 'cross_attn.to_q',
             'cross_attn.to_kv', 'cross_attn.to_out'],
            rank=ss_lora_rank if ss_lora_rank is not None else lora_rank,
            alpha=ss_lora_alpha if ss_lora_alpha is not None else lora_alpha,
            dropout=lora_dropout)
        state_dict = torch.load(sparse_structure_lora_path, map_location='cpu', weights_only=True)
        # `rope_phases` is a deterministic runtime buffer generated from the
        # model config.  Older LoRA checkpoints may contain a bfloat16 copy
        # (or a cast complex copy) which overwrites the freshly initialized
        # complex64 RoPE phases and makes structure decoding collapse to an
        # empty coordinate set.  It is not a trainable LoRA parameter.
        state_dict.pop('rope_phases', None)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            print(f"    [warning] SS LoRA checkpoint missing={len(missing)}, unexpected={len(unexpected)}")
        print(f"    SS LoRA layers: {len(replaced)}")
        loaded.append('sparse_structure_lora')
    elif sparse_structure_lora_path:
        print(f"  [跳过] Sparse Structure Flow LoRA 路径不存在: {sparse_structure_lora_path}")

    # --- Shape VAE Decoder ---
    if vae_decoder_path and os.path.exists(vae_decoder_path):
        print(f"  加载微调 VAE decoder: {vae_decoder_path}")
        state_dict = torch.load(vae_decoder_path, map_location='cpu')
        model = pipeline.models['shape_slat_decoder']
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"    [warning] missing keys: {len(missing)}")
        if unexpected:
            print(f"    [warning] unexpected keys: {len(unexpected)}")
        loaded.append('vae_decoder')
    else:
        print(f"  [跳过] VAE decoder 未找到或路径不存在")

    # --- Shape Flow 512 ---
    if flow_512_path and os.path.exists(flow_512_path):
        print(f"  加载微调 Shape Flow (512): {flow_512_path}")
        state_dict = torch.load(flow_512_path, map_location='cpu')
        model = pipeline.models['shape_slat_flow_model_512']
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"    [warning] missing keys: {len(missing)}")
        if unexpected:
            print(f"    [warning] unexpected keys: {len(unexpected)}")
        loaded.append('flow_512')
    else:
        print(f"  [跳过] Shape Flow (512) 未找到或路径不存在")

    # --- Shape Flow 512 LoRA ---
    # LoRA for the 512 flow: required by 1024_cascade/1536_cascade where the
    # 512 model is the LR stage that proposes the upsampled token set.
    if flow_512_lora_path and os.path.exists(flow_512_lora_path):
        print(f"  加载 Shape Flow (512) LoRA: {flow_512_lora_path}")
        model = pipeline.models['shape_slat_flow_model_512']
        replaced = apply_lora_to_linear_layers(
            model,
            ['self_attn.to_qkv', 'self_attn.to_out', 'cross_attn.to_q',
             'cross_attn.to_kv', 'cross_attn.to_out'],
            rank=shape_lora_rank if shape_lora_rank is not None else lora_rank,
            alpha=shape_lora_alpha if shape_lora_alpha is not None else lora_alpha,
            dropout=lora_dropout)
        state_dict = torch.load(flow_512_lora_path, map_location='cpu', weights_only=True)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            print(f"    [warning] LoRA checkpoint missing={len(missing)}, unexpected={len(unexpected)}")
        print(f"    LoRA layers: {len(replaced)}")
        loaded.append('flow_512_lora')
    elif flow_512_lora_path:
        print(f"  [跳过] Shape Flow (512) LoRA 路径不存在: {flow_512_lora_path}")

    # --- Shape Flow 1024 ---
    # 仅当显式提供 --flow_1024 时才替换；否则保持预训练权重。
    # （曾有的"512 权重移植到 1024"fallback 会导致 1024 精修阶段表面破碎，已移除）
    if flow_1024_path and os.path.exists(flow_1024_path):
        print(f"  加载微调 Shape Flow (1024): {flow_1024_path}")
        state_dict = torch.load(flow_1024_path, map_location='cpu')
        model = pipeline.models['shape_slat_flow_model_1024']
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"    [warning] missing keys: {len(missing)}")
        if unexpected:
            print(f"    [warning] unexpected keys: {len(unexpected)}")
        loaded.append('flow_1024')
    elif flow_1024_path:
        print(f"  [跳过] Shape Flow (1024) 路径不存在: {flow_1024_path}")

    # 1024 Flow LoRA: inject the same attention adapters used during training,
    # then load the adapter checkpoint (which also contains frozen base keys).
    if flow_1024_lora_path and os.path.exists(flow_1024_lora_path):
        print(f"  加载 Shape Flow (1024) LoRA: {flow_1024_lora_path}")
        model = pipeline.models['shape_slat_flow_model_1024']
        replaced = apply_lora_to_linear_layers(
            model,
            ['self_attn.to_qkv', 'self_attn.to_out', 'cross_attn.to_q',
             'cross_attn.to_kv', 'cross_attn.to_out'],
            rank=shape_lora_rank if shape_lora_rank is not None else lora_rank,
            alpha=shape_lora_alpha if shape_lora_alpha is not None else lora_alpha,
            dropout=lora_dropout)
        state_dict = torch.load(flow_1024_lora_path, map_location='cpu', weights_only=True)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            print(f"    [warning] LoRA checkpoint missing={len(missing)}, unexpected={len(unexpected)}")
        print(f"    LoRA layers: {len(replaced)}")
        loaded.append('flow_1024_lora')
    elif flow_1024_lora_path:
        print(f"  [跳过] Shape Flow (1024) LoRA 路径不存在: {flow_1024_lora_path}")

    # --- Texture SLat Flow (1024) LoRA ---
    # 同 shape flow LoRA 配方 (rank8/alpha8, 5 个 attention 目标), 挂在
    # tex_slat_flow_model_1024 上。只在显式提供 --tex_flow_lora 时生效。
    if tex_slat_lora_path and os.path.exists(tex_slat_lora_path):
        print(f"  加载 Texture SLat Flow (1024) LoRA: {tex_slat_lora_path}")
        model = pipeline.models['tex_slat_flow_model_1024']
        replaced = apply_lora_to_linear_layers(
            model,
            ['self_attn.to_qkv', 'self_attn.to_out', 'cross_attn.to_q',
             'cross_attn.to_kv', 'cross_attn.to_out'],
            rank=tex_lora_rank if tex_lora_rank is not None else lora_rank,
            alpha=tex_lora_alpha if tex_lora_alpha is not None else lora_alpha,
            dropout=lora_dropout)
        state_dict = torch.load(tex_slat_lora_path, map_location='cpu', weights_only=True)
        state_dict.pop('rope_phases', None)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            print(f"    [warning] Tex LoRA checkpoint missing={len(missing)}, unexpected={len(unexpected)}")
        print(f"    Tex LoRA layers: {len(replaced)}")
        loaded.append('tex_slat_lora')
    elif tex_slat_lora_path:
        print(f"  [跳过] Texture SLat Flow (1024) LoRA 路径不存在: {tex_slat_lora_path}")

    return loaded


def infer_single(image_path, pipeline, output_dir, seed, envmap=None, max_faces=16777216,
                 no_video=False, geometry_only=False, fast_glb=False):
    """处理单张图片"""
    name = os.path.splitext(os.path.basename(image_path))[0]
    print(f"\n  推理: {name}")

    image = Image.open(image_path)
    if image.mode != 'RGBA':
        image = image.convert('RGBA')

    mesh = pipeline.run(image, seed=seed)[0]
    mesh.simplify(max_faces)

    # Geometry-only export is deliberately UV-free.  For high-resolution
    # meshes xatlas chart generation can dominate inference time or stall;
    # PLY provides a fast, faithful shape comparison across pipeline types.
    if geometry_only:
        ply_path = os.path.join(output_dir, f'{name}.ply')
        trimesh.Trimesh(
            vertices=mesh.vertices.detach().cpu().numpy(),
            faces=mesh.faces.detach().cpu().numpy(),
            process=False,
        ).export(ply_path)
        print(f"    PLY (仅几何): {ply_path}")
        return None, ply_path

    if fast_glb:
        glb_path = os.path.join(output_dir, f'{name}.glb')
        trimesh.Trimesh(
            vertices=mesh.vertices.detach().cpu().numpy(),
            faces=mesh.faces.detach().cpu().numpy(),
            process=False,
        ).export(glb_path, file_type='glb')
        print(f"    GLB (快速无纹理): {glb_path}")
        return None, glb_path

    # 导出 GLB (先于视频: GLB 是主要产物，视频渲染耗时长且容易被中断)
    print(f"    导出 GLB...")
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices, faces=mesh.faces,
        attr_volume=mesh.attrs, coords=mesh.coords,
        attr_layout=mesh.layout, voxel_size=mesh.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=1000000, texture_size=4096,
        remesh=True, remesh_band=1, remesh_project=0, verbose=False,
    )
    glb_path = os.path.join(output_dir, f'{name}.glb')
    glb.export(glb_path)
    print(f"    GLB: {glb_path}")

    # 渲染视频
    if no_video:
        print(f"    跳过视频渲染 (--no_video)")
        return None, glb_path
    print(f"    渲染视频...")
    video = render_utils.make_pbr_vis_frames(
        render_utils.render_video(mesh, envmap=envmap)
    )
    video_path = os.path.join(output_dir, f'{name}.mp4')
    imageio.mimsave(video_path, video, fps=15)
    print(f"    视频: {video_path}")

    return video_path, glb_path


def main():
    parser = argparse.ArgumentParser(description='TRELLIS.2 微调模型推理')
    # 输入
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--image', type=str, help='单张输入图片路径')
    group.add_argument('--input_dir', type=str, help='输入图片目录 (支持 png/jpg/jpeg/webp)')

    # 输出
    parser.add_argument('--output', type=str, default='./inference_output',
                        help='输出目录 (默认: ./inference_output)')

    # 模型路径
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints',
                        help='预训练模型目录 (默认: ./checkpoints)')
    parser.add_argument('--vae_decoder', type=str, default=None,
                        help='VAE decoder checkpoint 路径 (默认: 自动查找 results/latest_vae)')
    parser.add_argument('--flow_512', type=str, default=None,
                        help='Shape Flow 512 checkpoint 路径 (默认: 自动查找 results/latest_flow)')
    parser.add_argument('--flow_512_lora', type=str, default=None,
                        help='Shape Flow 512 LoRA checkpoint (.pt)，1024_cascade/1536_cascade 的 LR 阶段用')
    parser.add_argument('--flow_1024', type=str, default=None,
                        help='Shape Flow 1024 checkpoint 路径 (默认: None，保持预训练权重)')
    parser.add_argument('--flow_1024_lora', type=str, default=None,
                        help='Shape Flow 1024 LoRA checkpoint (.pt)')
    parser.add_argument('--tex_flow_lora', type=str, default=None,
                        help='Texture SLat Flow (1024) LoRA checkpoint (.pt)')
    parser.add_argument('--sparse_structure_lora', type=str, default=None,
                        help='Sparse Structure Flow LoRA checkpoint (.pt)')
    parser.add_argument('--lora_rank', type=int, default=8)
    parser.add_argument('--lora_alpha', type=float, default=8.0)
    parser.add_argument('--ss_lora_rank', type=int, default=None)
    parser.add_argument('--ss_lora_alpha', type=float, default=None)
    parser.add_argument('--shape_lora_rank', type=int, default=None)
    parser.add_argument('--shape_lora_alpha', type=float, default=None)
    parser.add_argument('--tex_lora_rank', type=int, default=None)
    parser.add_argument('--tex_lora_alpha', type=float, default=None)
    parser.add_argument('--lora_dropout', type=float, default=0.0)

    # 模式
    parser.add_argument('--vae_only', action='store_true',
                        help='仅使用 VAE decoder，不替换 512 Flow (显式 --flow_1024 仍生效)')
    parser.add_argument('--pretrained_only', action='store_true',
                        help='仅使用官方预训练权重，不自动加载任何微调 checkpoint')
    parser.add_argument('--pipeline_type', type=str, default='1024_cascade',
                        choices=['512', '1024', '1024_cascade', '1536_cascade'],
                        help='Pipeline 类型 (默认: 1024_cascade)')
    parser.add_argument('--seed', type=int, default=42, help='随机种子 (默认: 42)')
    parser.add_argument('--ss_swap_yz', action='store_true',
                        help='把 SS 输出的坐标做 y/z 交换。本地 SS 监督缺少 y/z 对齐，'
                             '用本地训出的 SS LoRA 时成品的朝向依赖它')
    parser.add_argument('--ss_canonicalize', action='store_true',
                        help='按样本把 SS 占据体最长轴规范到下游期望的长度轴。'
                             'SS 采样朝向是双稳态的，全局交换只能修一半，'
                             '本开关对两个分支都有效（优先于 --ss_swap_yz）')
    parser.add_argument('--ss_cond_model', type=str, default='pipeline',
                        choices=['pipeline', 'dinov2'],
                        help='SS 阶段的条件编码器：pipeline=共用管线里的 DINOv3（默认）；'
                             'dinov2=用 DINOv2Local@518 + 1.2 裁剪（复刻旧 SS run 的训练口径）')
    parser.add_argument('--dinov2_path', type=str,
                        default='/data/lzx/ReconViaGen/checkpoints_dinov2')
    parser.add_argument('--no_video', action='store_true',
                        help='跳过视频渲染，仅导出 GLB (比较实验推荐，节省时间)')
    parser.add_argument('--geometry_only', action='store_true',
                        help='快速导出无 UV/纹理的 PLY，避免高面数 UV 展开阻塞')
    parser.add_argument('--fast_glb', action='store_true',
                        help='仅用于几何调试：导出无纹理 GLB；正式结果不要使用，会跳过 O-Voxel 重网格/坐标转换')
    parser.add_argument('--max_faces', type=int, default=16777216,
                        help='简化后最大面数 (默认: 16777216 = 2^24)')

    opt = parser.parse_args()

    if opt.pretrained_only and (opt.vae_only or opt.vae_decoder or opt.flow_512 or opt.flow_1024):
        parser.error('--pretrained_only 不能与 --vae_only、--vae_decoder、--flow_512 或 --flow_1024 同时使用')

    # ── 自动检测 latest 权重 ─────────────────────────────────
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if not opt.pretrained_only and opt.vae_decoder is None:
        ckpt_dir = os.path.join(project_dir, 'results', 'latest_vae', 'ckpts')
        candidates = [os.path.join(ckpt_dir, 'decoder_latest.pt')]
        if os.path.isdir(ckpt_dir):
            # 非 EMA 优先 (EMA 可能含大量预训练稀释)
            candidates += sorted(glob.glob(os.path.join(ckpt_dir, 'decoder_step*.pt')), reverse=True)
            candidates += sorted(glob.glob(os.path.join(ckpt_dir, 'decoder_ema*.pt')), reverse=True)
        for c in candidates:
            if os.path.exists(c):
                opt.vae_decoder = c
                break

    if not opt.pretrained_only and not opt.vae_only:
        # 显式传入 --flow_512_lora 时跳过 latest_flow 自动检测:
        # 旧的全量 512 检查点会覆盖 LoRA 的官方基线权重。
        if opt.flow_512 is None and opt.flow_512_lora is None:
            ckpt_dir = os.path.join(project_dir, 'results', 'latest_flow', 'ckpts')
            candidates = [os.path.join(ckpt_dir, 'denoiser_latest.pt')]
            if os.path.isdir(ckpt_dir):
                candidates += sorted(glob.glob(os.path.join(ckpt_dir, 'denoiser_ema*.pt')), reverse=True)
                candidates += sorted(glob.glob(os.path.join(ckpt_dir, 'denoiser_step*.pt')), reverse=True)
            for c in candidates:
                if os.path.exists(c):
                    opt.flow_512 = c
                    # 注意: 不自动设置 flow_1024。1024 精修模型必须保持预训练权重，
                    # 把 512 权重移植进 1024 会导致 1536_cascade 精修阶段表面破碎。
                    break

    # ── 加载 Pipeline ────────────────────────────────────────
    print("=" * 60)
    print("  TRELLIS.2 微调模型推理")
    print("=" * 60)

    print(f"\n加载预训练 pipeline from: {opt.checkpoints_dir}")
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(opt.checkpoints_dir)
    pipeline.default_pipeline_type = opt.pipeline_type
    pipeline.cuda()
    if getattr(opt, 'ss_canonicalize', False):
        pipeline.ss_canonicalize = True
        print('  SS 输出坐标: 逐样本最长轴规范化已开启')
    if getattr(opt, 'ss_swap_yz', False):
        pipeline.ss_swap_yz = True
        print('  SS 输出坐标: y/z 交换已开启')
    if getattr(opt, 'ss_cond_model', 'pipeline') == 'dinov2':
        from trellis2.trainers.flow_matching.mixins.image_conditioned import (
            DinoV2LocalFeatureExtractor)
        pipeline.ss_image_cond_model = DinoV2LocalFeatureExtractor(
            model_name=opt.dinov2_path, image_size=518)
        pipeline.ss_cond_resolution = 518
        pipeline.ss_cond_crop_scale = 1.2
        print('  SS 条件编码器: DINOv2Local@518 + crop_scale=1.2（旧训练口径）')
    print("  Pipeline 加载完成")

    # ── 替换微调权重 ─────────────────────────────────────────
    print(f"\n替换微调权重:")
    loaded = load_finetuned_weights(
        pipeline,
        vae_decoder_path=opt.vae_decoder,
        # --vae_only 只跳过 512 的自动检测；显式传入的 --flow_1024 始终生效，
        # 以支持 "微调 decoder + 预训练 512 + 微调 1024" 的组合。
        flow_512_path=None if opt.vae_only else opt.flow_512,
        flow_512_lora_path=None if opt.vae_only else opt.flow_512_lora,
        flow_1024_path=opt.flow_1024,
        flow_1024_lora_path=opt.flow_1024_lora,
        tex_slat_lora_path=opt.tex_flow_lora,
        sparse_structure_lora_path=opt.sparse_structure_lora,
        lora_rank=opt.lora_rank, lora_alpha=opt.lora_alpha,
        lora_dropout=opt.lora_dropout,
        ss_lora_rank=opt.ss_lora_rank, ss_lora_alpha=opt.ss_lora_alpha,
        shape_lora_rank=opt.shape_lora_rank, shape_lora_alpha=opt.shape_lora_alpha,
        tex_lora_rank=opt.tex_lora_rank, tex_lora_alpha=opt.tex_lora_alpha,
    )

    if not loaded:
        print("\n[警告] 没有加载任何微调权重，将使用纯预训练模型推理")
    else:
        print(f"\n  已加载: {', '.join(loaded)}")

    # ── 环境贴图 ─────────────────────────────────────────────
    envmap = EnvMap(torch.ones(256, 512, 3, dtype=torch.float32, device='cuda') * 0.5)

    # ── 推理 ─────────────────────────────────────────────────
    os.makedirs(opt.output, exist_ok=True)

    if opt.image:
        image_paths = [opt.image]
    else:
        extensions = ['*.png', '*.PNG', '*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.webp', '*.WEBP']
        image_paths = []
        for ext in extensions:
            image_paths.extend(glob.glob(os.path.join(opt.input_dir, ext)))
        image_paths = sorted(image_paths)
        print(f"\n找到 {len(image_paths)} 张图片")

    if not image_paths:
        print("[错误] 没有找到输入图片!")
        sys.exit(1)

    print(f"\n开始推理 ({len(image_paths)} 张)...")
    for i, img_path in enumerate(image_paths):
        print(f"\n[{i+1}/{len(image_paths)}]", end='')
        try:
            infer_single(img_path, pipeline, opt.output, seed=opt.seed, envmap=envmap,
                         max_faces=opt.max_faces, no_video=opt.no_video,
                         geometry_only=opt.geometry_only, fast_glb=opt.fast_glb)
        except Exception as e:
            print(f"\n  [错误] {img_path}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'=' * 60}")
    print(f"  完成! 输出目录: {opt.output}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()

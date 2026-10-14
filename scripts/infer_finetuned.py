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

安全边界:
    - 只读图片/权重；所有产物都写进 --output 目录，且该目录若已存在 run_manifest.json 会直接报错（防覆盖旧 run）。
    - 权重与图片路径在碰 GPU 之前先做存在性预检；LoRA 走 strict 校验（150 层），不匹配即失败。
    - 需要 CUDA；默认 CUDA_VISIBLE_DEVICES=7，显式设置则不覆盖。不会改动任何权重文件。
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
from scripts.utils.inference_contract import (TARGETS, require_file, HashCache, checked_load, validate_lora, glb_coordinates, export_metadata, timed, pipeline_timing, discover_images, run_batch, write_manifest, local_pipeline_weights)

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
    full = [('vae_decoder', 'shape_slat_decoder', vae_decoder_path),
            ('flow_512', 'shape_slat_flow_model_512', flow_512_path),
            ('flow_1024', 'shape_slat_flow_model_1024', flow_1024_path)]
    adapters = [('sparse_structure_lora','sparse_structure_flow_model',sparse_structure_lora_path,ss_lora_rank,ss_lora_alpha),
                ('flow_512_lora','shape_slat_flow_model_512',flow_512_lora_path,shape_lora_rank,shape_lora_alpha),
                ('flow_1024_lora','shape_slat_flow_model_1024',flow_1024_lora_path,shape_lora_rank,shape_lora_alpha),
                ('tex_slat_lora','tex_slat_flow_model_1024',tex_slat_lora_path,tex_lora_rank,tex_lora_alpha)]
    # Validate every requested path before mutating any model.
    for _,_,path in full:
        if path:require_file(path)
    for _,_,path,_,_ in adapters:
        if path:require_file(path)
    for name,key,path in full:
        if not path:continue
        print(f'  Loading {name}: {path}',flush=True)
        state=torch.load(path,map_location='cpu',weights_only=True)
        checked_load(pipeline.models[key],state,allow_rope=True)
        loaded.append(name)
    for name,key,path,rank,alpha in adapters:
        if not path:continue
        rank=lora_rank if rank is None else rank
        alpha=lora_alpha if alpha is None else alpha
        model=pipeline.models[key]
        print(f'  Loading {name}: {path} rank={rank} alpha={alpha}',flush=True)
        replaced=apply_lora_to_linear_layers(model,TARGETS,rank=rank,alpha=alpha,dropout=lora_dropout)
        state=torch.load(path,map_location='cpu',weights_only=True)
        # Runtime-derived RoPE must not be overwritten by historical cast buffers.
        state.pop('rope_phases',None)
        validate_lora(model,replaced,state,rank,alpha,lora_dropout)
        checked_load(model,state,allow_rope=True)
        print(f'  Loaded {name}: 150 verified adapters, eval mode',flush=True)
        loaded.append(name)
    return loaded



def infer_single(image_path, pipeline, output_dir, seed, envmap=None, max_faces=16777216,
                 no_video=False, geometry_only=False, fast_glb=False, timings=None):
    """处理单张图片：跑 pipeline -> 简化 -> 导出产物，返回 (video_path|None, glb/ply path)。
    geometry_only 出无 UV 的 .ply、fast_glb 出无纹理 .glb（都不算正式验收产物）；timings 记录各阶段耗时。
    """
    name = os.path.splitext(os.path.basename(image_path))[0]
    print(f"\n  推理: {name}")

    image = Image.open(image_path)
    if image.mode != 'RGBA':
        image = image.convert('RGBA')

    timings={} if timings is None else timings
    synchronize=torch.cuda.synchronize
    with pipeline_timing(pipeline,timings,synchronize):
        with timed(timings,'pipeline_run',synchronize):
            mesh = pipeline.run(image, seed=seed)[0]
    with timed(timings,'simplify',synchronize):
        mesh.simplify(max_faces)

    # Geometry-only export is deliberately UV-free.  For high-resolution
    # meshes xatlas chart generation can dominate inference time or stall;
    # PLY provides a fast, faithful shape comparison across pipeline types.
    if geometry_only:
        with timed(timings,'export',synchronize):
            ply_path = os.path.join(output_dir, f'{name}.ply')
            trimesh.Trimesh(
                vertices=glb_coordinates(mesh.vertices.detach().cpu().numpy()),
                faces=mesh.faces.detach().cpu().numpy(),
                process=False,
            ).export(ply_path)
            print(f"    PLY (仅几何): {ply_path}")
            return None, ply_path

    if fast_glb:
        with timed(timings,'export',synchronize):
            glb_path = os.path.join(output_dir, f'{name}.glb')
            trimesh.Trimesh(
                vertices=glb_coordinates(mesh.vertices.detach().cpu().numpy()),
                faces=mesh.faces.detach().cpu().numpy(),
                process=False,
            ).export(glb_path, file_type='glb')
            print(f"    GLB (快速无纹理): {glb_path}")
            return None, glb_path

    with timed(timings,'export',synchronize):
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


_ACTIVE_RUN = None

def main():
    """推理主流程：参数/路径预检 -> 建 output 与 run_manifest(v2) -> 自动探测 latest 权重 ->
    加载 pipeline、替换微调权重 -> 逐图推理（每图失败都记进 manifest 并让退出码非 0）。
    """
    global _ACTIVE_RUN
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
                        help='几何调试无纹理 GLB；统一正式GLB坐标，但跳过remesh/UV，不能作为正式验收')
    parser.add_argument('--max_faces', type=int, default=16777216,
                        help='简化后最大面数 (默认: 16777216 = 2^24)')

    parser.add_argument('--image_timeout_seconds',type=float,default=0,
                        help='Unix Python alarm per image; CUDA C extensions may delay it; use outer subprocess deadline')
    opt = parser.parse_args()
    import math
    if not math.isfinite(opt.image_timeout_seconds) or opt.image_timeout_seconds<0:parser.error('timeout must be finite nonnegative')
    if opt.geometry_only and opt.fast_glb:parser.error('geometry_only and fast_glb are mutually exclusive')
    explicit_fields=['vae_decoder','flow_512','flow_512_lora','flow_1024','flow_1024_lora','tex_flow_lora','sparse_structure_lora']
    explicit={k:getattr(opt,k) for k in explicit_fields if getattr(opt,k) is not None}
    if opt.pretrained_only and (opt.vae_only or explicit):parser.error('pretrained_only cannot combine with any explicit fine-tuned weights')
    if opt.vae_only and (opt.flow_512 or opt.flow_512_lora):parser.error('vae_only conflicts with explicit 512 flow weights')
    # All argument/input failures precede pipeline construction or GPU allocation.
    # 所有参数/输入校验都放在构建 pipeline 与申请显存之前。
    for rank in [opt.lora_rank,opt.ss_lora_rank,opt.shape_lora_rank,opt.tex_lora_rank]:
        if rank is not None and rank<=0:parser.error('LoRA rank must be positive')
    for alpha in [opt.lora_alpha,opt.ss_lora_alpha,opt.shape_lora_alpha,opt.tex_lora_alpha]:
        if alpha is not None and (not math.isfinite(alpha) or alpha<=0):parser.error('LoRA alpha must be finite positive')
    if not math.isfinite(opt.lora_dropout) or not 0<=opt.lora_dropout<1:parser.error('LoRA dropout must be finite 0..1')
    if opt.max_faces<=0:parser.error('max_faces must be positive')
    for value in explicit.values():require_file(value)
    image_paths=discover_images(opt.image,opt.input_dir)
    os.makedirs(opt.output,exist_ok=True)
    manifest_path=os.path.join(opt.output,'run_manifest.json')
    if os.path.exists(manifest_path):raise FileExistsError('existing run_manifest.json: use a fresh output directory')
    hashes=HashCache()
    # 先落盘 status=loading 的骨架 manifest，再用真实权重哈希逐次原子更新（失败也能留下证据）。
    manifest={'schema':'trellis.inference-run.v2','status':'loading','flags':vars(opt).copy(),
              'seed':opt.seed,'pipeline_type':opt.pipeline_type,'export':export_metadata(opt.geometry_only,opt.fast_glb),
              'explicit_weights':{k:hashes.record(v) for k,v in explicit.items()},'images':[],
              'lora':{'targets':list(TARGETS),'expected_layers':150,'rank':opt.lora_rank,'alpha':opt.lora_alpha,
                      'ss_rank':opt.ss_lora_rank or opt.lora_rank,'ss_alpha':opt.lora_alpha if opt.ss_lora_alpha is None else opt.ss_lora_alpha,
                      'shape_rank':opt.shape_lora_rank or opt.lora_rank,'shape_alpha':opt.lora_alpha if opt.shape_lora_alpha is None else opt.shape_lora_alpha,
                      'tex_rank':opt.tex_lora_rank or opt.lora_rank,'tex_alpha':opt.lora_alpha if opt.tex_lora_alpha is None else opt.tex_lora_alpha,
                      'alpha_verification':'historical checkpoints lack alpha metadata; CLI value recorded, not inferred','dropout':opt.lora_dropout}}
    _ACTIVE_RUN=(manifest_path,manifest)
    manifest['created_utc']=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()
    write_manifest(manifest_path,manifest)

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

    manifest['resolved_weights']={k:hashes.record(getattr(opt,k)) for k in explicit_fields if getattr(opt,k)}
    manifest['base_weights']=local_pipeline_weights(opt.checkpoints_dir,hashes)
    manifest['automatic_weights']=[k for k in explicit_fields if getattr(opt,k) and k not in explicit]
    manifest['official_components_retained']=[k for k in explicit_fields if not getattr(opt,k)]
    write_manifest(manifest_path,manifest)

    # ── 加载 Pipeline ────────────────────────────────────────
    print("=" * 60)
    print("  TRELLIS.2 微调模型推理")
    print("=" * 60)

    print(f"\n加载预训练 pipeline from: {opt.checkpoints_dir}")
    manifest['load_timings']={}
    with timed(manifest['load_timings'],'pipeline_load'):
        pipeline = Trellis2ImageTo3DPipeline.from_pretrained(opt.checkpoints_dir)
    manifest['actual_base_sources']={}
    for name,model in pipeline.models.items():
        source=getattr(model,'_checkpoint_source',None)
        manifest['actual_base_sources'][name]=({'requested':source.get('requested'),
            'config':hashes.record(source['config_file']),'weights':hashes.record(source['model_file']),
            'computed_buffers_regenerated':source.get('computed_buffers_regenerated',[])} if source else {'status':'source_metadata_unavailable'})
    write_manifest(manifest_path,manifest)
    pipeline.default_pipeline_type = opt.pipeline_type
    with timed(manifest['load_timings'],'pipeline_to_cuda',torch.cuda.synchronize):
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

    manifest['loaded_components']=loaded
    manifest['actual_coordinate_flags']={'ss_canonicalize':bool(pipeline.ss_canonicalize),'ss_swap_yz':bool(pipeline.ss_swap_yz)}
    # 开过坐标修正或 raw 导出就不算正式验收；evaluate_fixed_frame 据此拒绝。
    manifest['acceptance_flags_eligible']=not(pipeline.ss_canonicalize or pipeline.ss_swap_yz or opt.geometry_only or opt.fast_glb)
    manifest['samplers']={k:getattr(pipeline,k,None) for k in ['sparse_structure_sampler_params','shape_slat_sampler_params','tex_slat_sampler_params']}
    write_manifest(manifest_path,manifest)
    if not loaded:
        print("\n[警告] 没有加载任何微调权重，将使用纯预训练模型推理")
    else:
        print(f"\n  已加载: {', '.join(loaded)}")

    # ── 环境贴图 ─────────────────────────────────────────────
    envmap = None if opt.no_video or opt.geometry_only or opt.fast_glb else EnvMap(torch.ones(256, 512, 3, dtype=torch.float32, device='cuda') * 0.5)

    # A failed image is recorded and any partial failure returns nonzero.
    # 单图失败会被记进 manifest，最终只要有失败就返回非 0（批量跑不会被静默吞掉）。
    def runner(path,timings):
        return infer_single(path,pipeline,opt.output,seed=opt.seed,envmap=envmap,
                            max_faces=opt.max_faces,no_video=opt.no_video,
                            geometry_only=opt.geometry_only,fast_glb=opt.fast_glb,timings=timings)
    code=run_batch(image_paths,runner,manifest,manifest_path,opt.image_timeout_seconds,hashes=hashes)
    print(f"Inference {'FAILED/PARTIAL' if code else 'SUCCESS'}: {manifest['successful_images']} success, {manifest['failed_images']} failed. Output: {opt.output}",flush=True)
    return code


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as e:
        # 任何未捕获异常都要把 run_manifest 收尾成 failed 并落盘，再以非 0 退出。
        if _ACTIVE_RUN is not None:
            path,manifest=_ACTIVE_RUN
            manifest.update(status='failed',run_error=f'{type(e).__name__}: {e}',finished_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat())
            write_manifest(path,manifest)
        print(f'[RUN FAILED] {type(e).__name__}: {e}',flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)

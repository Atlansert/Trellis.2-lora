"""批量推理脚本：把 ../single_side/ 下的所有 PNG 逐个转成 3D。

流程：加载本地 checkpoints 的 pipeline → 逐张图 run(pipeline_type='1024')
→ 渲染转台视频（mp4）→ 导出 GLB，产物与输入图同目录。
"""
import os
# 指定使用的 GPU，按需修改
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import imageio, torch, glob
from PIL import Image
from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.utils import render_utils
from trellis2.renderers import EnvMap
import o_voxel

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 输入目录：脚本所在目录的上一级 single_side/
INPUT_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, '..', 'single_side'))
# 渲染用环境贴图：这里用 0.5 的常量灰环境（非 HDRI 文件）
envmap = EnvMap(torch.ones(256, 512, 3, dtype=torch.float32, device='cuda') * 0.5)

print('Loading model...')
# 从本地 checkpoints 目录加载（此处不是 Hugging Face 仓库名）
pipeline = Trellis2ImageTo3DPipeline.from_pretrained(os.path.join(SCRIPT_DIR, 'checkpoints'))
pipeline.cuda()
print('Model loaded!')

# 读取 ../single_side/ 下所有 PNG，忽略其他文件
images = sorted(glob.glob(os.path.join(INPUT_DIR, '*.png')) +
                glob.glob(os.path.join(INPUT_DIR, '*.PNG')))
if not images:
    print(f'[错误] {INPUT_DIR} 中没有 PNG 文件!')
    exit(1)

print(f'找到 {len(images)} 张 PNG 图片')

for img_path in images:
    # 逐张图：推理 → 渲染视频 → 导出 GLB，产物名取输入文件名
    name = os.path.splitext(os.path.basename(img_path))[0]

    print(f'Processing {name}...')
    image = Image.open(img_path)
    mesh = pipeline.run(image, pipeline_type='1024')[0]
    # 简化网格以适配 nvdiffrast 的索引上限
    mesh.simplify(16777216)

    print(f'  Rendering video...')
    video = render_utils.make_pbr_vis_frames(render_utils.render_video(mesh, envmap=envmap))
    imageio.mimsave(os.path.join(INPUT_DIR, f'{name}.mp4'), video, fps=15)

    print(f'  Exporting GLB...')
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices, faces=mesh.faces,
        attr_volume=mesh.attrs, coords=mesh.coords,
        attr_layout=mesh.layout, voxel_size=mesh.voxel_size,
        aabb=[[-0.5,-0.5,-0.5],[0.5,0.5,0.5]],
        decimation_target=1000000, texture_size=4096,
        remesh=True, remesh_band=1, remesh_project=0, verbose=True,
    )
    glb.export(os.path.join(INPUT_DIR, f'{name}.glb'))
    print(f'  Saved {INPUT_DIR}/{name}.glb + .mp4')

print('All done!')




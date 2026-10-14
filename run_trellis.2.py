import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import imageio, torch, glob
from PIL import Image
from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.utils import render_utils
from trellis2.renderers import EnvMap
import o_voxel

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, '..', 'single_side'))
envmap = EnvMap(torch.ones(256, 512, 3, dtype=torch.float32, device='cuda') * 0.5)

print('Loading model...')
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
    name = os.path.splitext(os.path.basename(img_path))[0]

    print(f'Processing {name}...')
    image = Image.open(img_path)
    mesh = pipeline.run(image, pipeline_type='1024')[0]
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




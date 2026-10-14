"""
Blender script for rendering 24 views (every 15°) of a 3D model.
Runs inside Blender Python (called by render_views.py, not directly).

Usage (internal):
    blender -b -P render_view.py -- --object model.glb --output_dir /tmp/views

Output:
    {output_dir}/
        000_cam_000.png   # yaw=0°
        001_cam_015.png   # yaw=15°
        002_cam_030.png   # yaw=30°
        ...
        023_cam_345.png   # yaw=345°
        transforms.json   # camera parameters

中文说明：
- 用途：在 Blender 内部渲染 24 个视角（每 15°）的 RGBA 图，并导出相机参数。
- 调用方式（内部）：blender -b -P render_view.py -- --object model.glb --output_dir /tmp/views
  [--resolution 1024] [--radius 2.2] [--fov 30] [--transparent] [--gpu N]；由 render_views.py 等外部脚本调用。
- 输出：{output_dir}/<idx>_cam_<yaw>.png（000_cam_000.png … 023_cam_345.png）与 transforms.json
  （含 scale/offset/object_path 与逐帧 yaw_deg、camera_angle_x、transform_matrix）。
- 关键约定：相机在 **XY 平面环绕**（pitch 恒为 0）、以世界 **+Z 为竖直**，且相机 local Y 对齐世界竖直轴——
  这是下游体素化/轴置换所依赖的渲染帧约定；模型先归一化到 [-0.5,0.5]³ 并居中到原点。
- 色彩管理显式钉死 Filmic + sRGB（避免 Blender 4.x 默认 AgX 造成色调偏移）。
"""
import argparse, sys, os, math, json
import bpy
from mathutils import Vector
import numpy as np

IMPORT_FUNCTIONS = {
    "obj": bpy.ops.import_scene.obj,
    "glb": bpy.ops.import_scene.gltf,
    "gltf": bpy.ops.import_scene.gltf,
    "fbx": bpy.ops.import_scene.fbx,
    "stl": bpy.ops.import_mesh.stl,
    "ply": bpy.ops.import_mesh.ply,
}


def init_scene():
    """Clear all objects, materials, textures, images from the scene."""
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for mat in list(bpy.data.materials):
        bpy.data.materials.remove(mat, do_unlink=True)
    for tex in list(bpy.data.textures):
        bpy.data.textures.remove(tex, do_unlink=True)
    for img in list(bpy.data.images):
        bpy.data.images.remove(img, do_unlink=True)


def init_render(engine='CYCLES', resolution=1024, gpu=None, transparent=False):
    """Configure Cycles renderer: GPU 7, 32 samples, denoising, white background."""
    s = bpy.context.scene.render
    s.engine = engine
    s.resolution_x = resolution
    s.resolution_y = resolution
    s.resolution_percentage = 100
    s.image_settings.file_format = 'PNG'
    s.image_settings.color_mode = 'RGBA'
    s.film_transparent = transparent

    # 色彩管理必须与 data_for_train 既有渲染保持一致：Blender 3.x 默认 Filmic，
    # Blender 4.x 默认 AgX，会把白底从 0.806 压到 0.757、整体色调偏移约 4%。
    # 实测（同一对象 yaw=0）与存档渲染的逐像素差：Filmic 均值 0.74/255
    # （91% 像素 ≤2），AgX 均值 9.88/255。故此处显式钉死 Filmic。
    bpy.context.scene.display_settings.display_device = 'sRGB'
    bpy.context.scene.view_settings.view_transform = 'Filmic'
    bpy.context.scene.view_settings.look = 'None'
    bpy.context.scene.view_settings.exposure = 0.0
    bpy.context.scene.view_settings.gamma = 1.0

    # FIXED: use bpy.context.scene.cycles (Blender 3.0 API), not s.cycles
    cycles = bpy.context.scene.cycles
    cycles.device = 'GPU'
    cycles.samples = 32
    cycles.filter_type = 'BOX'
    cycles.filter_width = 1
    cycles.diffuse_bounces = 1
    cycles.glossy_bounces = 1
    cycles.transparent_max_bounces = 3
    cycles.transmission_bounces = 3
    cycles.use_denoising = True

    # Select CUDA GPU 7
    prefs = bpy.context.preferences.addons['cycles'].preferences
    prefs.get_devices()
    prefs.compute_device_type = 'CUDA'
    for i, device in enumerate(prefs.devices):
        device.use = (i == gpu) if gpu is not None else (i == 7)
        print(f'[Blender] CUDA device {i}: {device.name} -> {"ON" if device.use else "OFF"}')


# 相机始终注视原点：负 Z 轴指向目标，且相机 local Y（画面竖直方向）对齐世界 Z 轴。
def init_camera():
    """Create a camera that always looks at origin."""
    cam = bpy.data.objects.new('Camera', bpy.data.cameras.new('Camera'))
    bpy.context.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    cam.data.sensor_height = cam.data.sensor_width = 32
    cam_constraint = cam.constraints.new(type='TRACK_TO')
    cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
    cam_constraint.up_axis = 'UP_Y'
    cam_empty = bpy.data.objects.new("Empty", None)
    cam_empty.location = (0, 0, 0)
    bpy.context.scene.collection.objects.link(cam_empty)
    cam_constraint.target = cam_empty
    return cam


def init_white_lighting():
    """Pure white environment lighting — no directional lights, clean background."""
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.select_by_type(type="LIGHT")
    bpy.ops.object.delete()
    if bpy.context.scene.world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    else:
        world = bpy.context.scene.world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    for node in list(nodes):
        nodes.remove(node)
    bg = nodes.new(type="ShaderNodeBackground")
    bg.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    bg.inputs["Strength"].default_value = 1.0
    out = nodes.new(type="ShaderNodeOutputWorld")
    links.new(bg.outputs["Background"], out.inputs["Surface"])


def load_object(path):
    """Load a 3D model file (GLB, OBJ, FBX, PLY, STL).

    Blender 4.x removed ``bpy.ops.import_scene.obj`` in favour of
    ``bpy.ops.wm.obj_import``; pick whichever the running build provides so the
    same script works on 3.x and 4.x.
    """
    ext = path.split(".")[-1].lower()
    if ext not in IMPORT_FUNCTIONS:
        raise ValueError(f"Unsupported format: {ext}")
    if ext == "obj" and hasattr(bpy.ops.wm, "obj_import"):
        bpy.ops.wm.obj_import(filepath=path)
    else:
        import_func = IMPORT_FUNCTIONS[ext]
        if ext in ("glb", "gltf"):
            import_func(filepath=path, merge_vertices=True, import_shading='NORMALS')
        else:
            import_func(filepath=path)
    print(f'[Blender] Loaded: {path}')


def scene_bbox():
    """Compute bounding box of all mesh objects in the scene."""
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, bpy.types.Mesh):
            found = True
            for coord in obj.bound_box:
                coord = obj.matrix_world @ Vector(coord)
                bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
                bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    if not found:
        raise RuntimeError("no mesh objects in scene")
    return Vector(bbox_min), Vector(bbox_max)


# 归一化：按最长边缩放使模型落在 [-0.5,0.5]³，再把包围盒中心平移到原点（与其它流水线约定一致）。
def normalize_scene():
    """Scale and translate scene so model fits in [-0.5, 0.5]³ centered at origin."""
    roots = [obj for obj in bpy.context.scene.objects.values() if not obj.parent]
    if not roots:
        raise RuntimeError("场景中无物体 — GLB 文件可能为空或损坏")
    if len(roots) > 1:
        parent = bpy.data.objects.new("ParentEmpty", None)
        bpy.context.scene.collection.objects.link(parent)
        for obj in roots:
            obj.parent = parent
    else:
        parent = roots[0]
    bbox_min, bbox_max = scene_bbox()
    scale = 1 / max(bbox_max - bbox_min)
    parent.scale = parent.scale * scale
    bpy.context.view_layer.update()
    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
    parent.matrix_world.translation += offset
    bpy.ops.object.select_all(action="DESELECT")
    return float(scale), offset


def get_transform_matrix(obj):
    """Extract 4x4 world transform matrix from a Blender object."""
    pos, rt, _ = obj.matrix_world.decompose()
    rt = rt.to_matrix()
    m = []
    for ii in range(3):
        m.append([rt[ii][jj] for jj in range(3)] + [pos[ii]])
    m.append([0, 0, 0, 1])
    return m


# 在 Blender 内运行：只解析 `--` 之后的参数（之前的属于 Blender 自身）。
def main():
    argv = sys.argv[sys.argv.index("--") + 1:]
    parser = argparse.ArgumentParser()
    parser.add_argument('--object', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--resolution', type=int, default=1024)
    parser.add_argument('--radius', type=float, default=2.2)
    parser.add_argument('--fov', type=float, default=30.0)
    parser.add_argument('--transparent', action='store_true',
                        help='透明背景渲染（alpha 通道即物体遮罩，与官方/推理约定一致）')
    parser.add_argument('--gpu', type=int, default=None,
                        help='CUDA 设备序号（可见设备内的下标）；缺省沿用旧行为 i==7')
    args = parser.parse_args(argv)

    os.makedirs(args.output_dir, exist_ok=True)

    # Setup
    init_scene()
    load_object(args.object)
    scale, offset = normalize_scene()
    print(f'[Blender] Normalized: scale={scale:.4f}')

    cam = init_camera()
    init_white_lighting()
    init_render(resolution=args.resolution, gpu=args.gpu, transparent=args.transparent)

    # 24 views every 15° (0°, 15°, 30°, ..., 345°)
    # 中文：相机在 XY 平面环绕（pitch=0），绕世界 Z 轴每 15° 一个视角，共 24 个。
    yaws_deg = list(range(0, 360, 15))
    labels = [f'cam_{yaw:03d}' for yaw in yaws_deg]

    to_export = {
        "scale": scale,
        "offset": [offset.x, offset.y, offset.z],
        "object_path": args.object,
        "frames": []
    }

    for i, (yaw_deg, label) in enumerate(zip(yaws_deg, labels)):
        yaw = math.radians(yaw_deg)
        pitch = 0.0
        cam_dir = np.array([
            math.cos(yaw) * math.cos(pitch),
            math.sin(yaw) * math.cos(pitch),
            math.sin(pitch)
        ])
        cam.location = (
            args.radius * cam_dir[0],
            args.radius * cam_dir[1],
            args.radius * cam_dir[2]
        )
        # 由水平视场角换算焦距：sensor 宽 32（半宽 16），lens = 16 / tan(fov/2)。
        cam.data.lens = 16 / math.tan(math.radians(args.fov) / 2)

        filepath = os.path.join(args.output_dir, f'{i:03d}_{label}.png')
        bpy.context.scene.render.filepath = filepath
        bpy.ops.render.render(write_still=True)
        bpy.context.view_layer.update()

        to_export["frames"].append({
            "file_path": os.path.basename(filepath),
            "yaw_label": label,
            "yaw_deg": yaw_deg,
            "camera_angle_x": math.radians(args.fov),
            "transform_matrix": get_transform_matrix(cam)
        })
        print(f'[Blender] Rendered ({i+1}/24): {os.path.basename(filepath)}')

    with open(os.path.join(args.output_dir, 'transforms.json'), 'w') as f:
        json.dump(to_export, f, indent=2)
    print('[Blender] Done.')


if __name__ == '__main__':
    main()

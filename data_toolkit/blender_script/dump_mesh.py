"""在 Blender 内部把模型导出成 mesh dump（pickle）。

- 用途：被 data_toolkit/dump_mesh.py 调用：加载模型 → 归一化到 [-0.5,0.5]³ → 导出分部件顶点/面。
- 调用方式：blender -b -P dump_mesh.py -- --object model.glb --output_path /tmp/out.pickle。
- 输出：pickle（{'objects': [{'vertices': float32 [N,3], 'faces': int32 [F,3]}, ...]}，已三角化，
  顶点已乘 obj.matrix_world 到世界坐标）；脚本失败时由调用方约定写 *_error.txt。
- 关键约定：几何取 evaluated depsgraph（含修改器/形态键结果）；只导出 MESH 类型对象；
  .blend 输入走“删掉隐藏对象”路径而不重新导入；导入函数按 Blender 3.x/4.x 版本自动适配。
"""
import argparse, sys, os, math, io
from typing import *
import bpy
import bmesh
from mathutils import Vector, Matrix
import numpy as np
import pickle


"""=============== BLENDER ==============="""

IMPORT_FUNCTIONS: Dict[str, Callable] = {
    "obj": bpy.ops.import_scene.obj if bpy.app.version[0] < 4 else bpy.ops.wm.obj_import,
    "glb": bpy.ops.import_scene.gltf,
    "gltf": bpy.ops.import_scene.gltf,
    "usd": bpy.ops.import_scene.usd,
    "fbx": bpy.ops.import_scene.fbx,
    "stl": bpy.ops.import_mesh.stl if bpy.app.version[0] < 4 else bpy.ops.wm.stl_import,
    "usda": bpy.ops.import_scene.usda,
    "dae": bpy.ops.wm.collada_import,
    "ply": bpy.ops.import_mesh.ply if bpy.app.version[0] < 4 else bpy.ops.wm.ply_import,
    "abc": bpy.ops.wm.alembic_import,
    "blend": bpy.ops.wm.append,
}


def init_scene() -> None:
    """Resets the scene to a clean state.

    Returns:
        None
    """
    # delete everything
    for obj in bpy.data.objects:
        bpy.data.objects.remove(obj, do_unlink=True)

    # delete all the materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)

    # delete all the textures
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)

    # delete all the images
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)


def load_object(object_path: str) -> None:
    """Loads a model with a supported file extension into the scene.

    Args:
        object_path (str): Path to the model file.

    Raises:
        ValueError: If the file extension is not supported.

    Returns:
        None
    """
    file_extension = object_path.split(".")[-1].lower()
    if file_extension is None:
        raise ValueError(f"Unsupported file type: {object_path}")

    if file_extension == "usdz":
        # install usdz io package
        dirname = os.path.dirname(os.path.realpath(__file__))
        usdz_package = os.path.join(dirname, "io_scene_usdz.zip")
        bpy.ops.preferences.addon_install(filepath=usdz_package)
        # enable it
        addon_name = "io_scene_usdz"
        bpy.ops.preferences.addon_enable(module=addon_name)
        # import the usdz
        from io_scene_usdz.import_usdz import import_usdz

        import_usdz(context, filepath=object_path, materials=True, animations=True)
        return None

    # load from existing import functions
    import_function = IMPORT_FUNCTIONS[file_extension]

    print(f"Loading object from {object_path}")
    if file_extension == "blend":
        import_function(directory=object_path, link=False)
    elif file_extension in {"glb", "gltf"}:
        import_function(filepath=object_path, merge_vertices=True, import_shading='NORMALS', bone_heuristic='TEMPERANCE')
    else:
        import_function(filepath=object_path)
        
        
def delete_invisible_objects() -> None:
    """Deletes all invisible objects in the scene.

    Returns:
        None
    """
    to_remove = []
    for obj in bpy.context.scene.objects:
        if obj.hide_viewport or obj.hide_render:
            obj.hide_viewport = False
            obj.hide_render = False
            obj.hide_select = False
            to_remove.append(obj)
    for obj in to_remove:
        bpy.data.objects.remove(obj, do_unlink=True)

    # Delete invisible collections
    invisible_collections = [col for col in bpy.data.collections if col.hide_viewport]
    for col in invisible_collections:
        bpy.data.collections.remove(col)
      

def scene_bbox() -> Tuple[Vector, Vector]:
    """Returns the bounding box of the scene.

    Taken from Shap-E rendering script
    (https://github.com/openai/shap-e/blob/main/shap_e/rendering/blender/blender_script.py#L68-L82)

    Returns:
        Tuple[Vector, Vector]: The minimum and maximum coordinates of the bounding box.
    """
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    scene_meshes = [obj for obj in bpy.context.scene.objects.values() if isinstance(obj.data, bpy.types.Mesh)]
    for obj in scene_meshes:
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")
    return Vector(bbox_min), Vector(bbox_max)


def normalize_scene() -> Tuple[float, Vector]:
    """Normalizes the scene by scaling and translating it to fit in a unit cube centered
    at the origin.

    Mostly taken from the Point-E / Shap-E rendering script
    (https://github.com/openai/point-e/blob/main/point_e/evals/scripts/blender_script.py#L97-L112),
    but fix for multiple root objects: (see bug report here:
    https://github.com/openai/shap-e/pull/60).

    Returns:
        Tuple[float, Vector]: The scale factor and the offset applied to the scene.
    """
    scene_root_objects = [obj for obj in bpy.context.scene.objects.values() if not obj.parent]
    if len(scene_root_objects) > 1:
        # create an empty object to be used as a parent for all root objects
        scene = bpy.data.objects.new("ParentEmpty", None)
        bpy.context.scene.collection.objects.link(scene)

        # parent all root objects to the empty object
        for obj in scene_root_objects:
            obj.parent = scene
    else:
        scene = scene_root_objects[0]

    bbox_min, bbox_max = scene_bbox()
    scale = 1 / max(bbox_max - bbox_min)
    scene.scale = scene.scale * scale

    # Apply scale to matrix_world.
    bpy.context.view_layer.update()
    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
    scene.matrix_world.translation += offset
    
    return scale, offset


# 主流程：初始化场景（或复用 .blend）→ 归一化 → 逐对象取求值后网格、三角化并导出 pickle。
def main(arg):    
    # Initialize context
    if arg.object.endswith(".blend"):
        delete_invisible_objects()
    else:
        init_scene()
        load_object(arg.object)
    print('[INFO] Scene initialized.')
    
    # Normalize scene
    scale, offset = normalize_scene()
    print('[INFO] Scene normalized.')
    
    # Start dumping
    depsgraph = bpy.context.evaluated_depsgraph_get()
    scene = bpy.context.scene
    output = {
        'objects': [],
    }

    # Dumping meshes
    for obj in scene.objects:
        if obj.type != 'MESH':
            continue
        
        pack = {
            "vertices": None,
            "faces": None,
        }
        
        eval_obj = obj.evaluated_get(depsgraph)
        eval_mesh = eval_obj.to_mesh()
        
        bm = bmesh.new()
        bm.from_mesh(eval_mesh)
        # 先把网格顶点变换到世界坐标，再三角化，保证导出结果与场景朝向一致。
        bm.transform(obj.matrix_world)
        bmesh.ops.triangulate(bm, faces=bm.faces)
        bm.to_mesh(eval_mesh)
        bm.free()
                
        pack["vertices"] = np.array([
            v.co[:] for v in eval_mesh.vertices
        ], dtype=np.float32)   # (N, 3)
        
        pack["faces"] = np.array([
            [eval_mesh.loops[i].vertex_index for i in poly.loop_indices]
            for poly in eval_mesh.polygons
        ], dtype=np.int32)   # (F, 3)

        output['objects'].append(pack)

    # Save output
    os.makedirs(os.path.dirname(arg.output_path), exist_ok=True)
    with open(arg.output_path, 'wb') as f:
        pickle.dump(output, f)
    print('[INFO] Output saved to {}.'.format(arg.output_path))

        
# 在 Blender 内运行：只解析 `--` 之后的参数（之前的属于 Blender 自身）。
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Renders given obj file by rotation a camera around it.')
    parser.add_argument('--object', type=str, help='Path to the 3D model file to be rendered.')
    parser.add_argument('--output_path', type=str, default='/tmp', help='The path the output will be dumped to.')
    argv = sys.argv[sys.argv.index("--") + 1:]
    args = parser.parse_args(argv)

    main(args)
    
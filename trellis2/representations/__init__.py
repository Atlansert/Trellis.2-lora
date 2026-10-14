"""几何/外观表示包：Mesh（可带 voxel 属性或 PBR 材质）与 Voxel 的懒加载入口。

__attributes 把类名映射到子模块，属性首次被访问时才真正 import。
"""
import importlib

# 类名 -> 子模块名
__attributes = {
    'Mesh': 'mesh',
    'Voxel': 'voxel',
    'MeshWithVoxel': 'mesh',
    'MeshWithPbrMaterial': 'mesh',
}

__submodules = []

__all__ = list(__attributes.keys()) + __submodules

def __getattr__(name):
    if name not in globals():
        if name in __attributes:
            module_name = __attributes[name]
            module = importlib.import_module(f".{module_name}", __name__)
            globals()[name] = getattr(module, name)
        elif name in __submodules:
            module = importlib.import_module(f".{name}", __name__)
            globals()[name] = module
        else:
            raise AttributeError(f"module {__name__} has no attribute {name}")
    return globals()[name]


# For Pylance
if __name__ == '__main__':
    from .mesh import Mesh, MeshWithVoxel, MeshWithPbrMaterial
    from .voxel import Voxel

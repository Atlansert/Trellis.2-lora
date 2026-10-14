"""datasets 包入口：按需（惰性）导出各数据集类。

职责：在 __attributes 中登记「类名 -> 相对子模块路径」，首次访问属性时才 import，
避免一次性加载所有数据集（及其对 models/渲染器等重依赖）。
输入：`from trellis2.datasets import XXXDataset` 之类的属性访问。
输出：对应的数据集类，或子模块本身。
契约：新增数据集需同时登记到 __attributes；类名必须与目标模块内定义的名字一致。
"""

import importlib

# 惰性导入表：键为对外暴露的名字，值为相对本包的子模块路径
__attributes = {
    'FlexiDualGridDataset': 'flexi_dual_grid',
    'SparseVoxelPbrDataset':'sparse_voxel_pbr',
    
    'SparseStructureLatent': 'sparse_structure_latent',
    'TextConditionedSparseStructureLatent': 'sparse_structure_latent',
    'ImageConditionedSparseStructureLatent': 'sparse_structure_latent',
    'ImageConditionedSparseStructureLatentViews': 'sparse_structure_latent_views',
    
    'SLat': 'structured_latent',
    'ImageConditionedSLat': 'structured_latent',
    'SLatShape': 'structured_latent_shape',
    'ImageConditionedSLatShape': 'structured_latent_shape',
    'MultiImageConditionedSLatShape': 'structured_latent_shape',
    'SLatPbr': 'structured_latent_svpbr',
    'ImageConditionedSLatPbr': 'structured_latent_svpbr',
}

# 以子模块形式导出的名字（当前为空）
__submodules = []

__all__ = list(__attributes.keys()) + __submodules

def __getattr__(name):
    # PEP 562 模块级 __getattr__：首次访问时才真正 import 并缓存进 globals()
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


# For Pylance（仅供静态检查/直接运行本文件时补全符号）
if __name__ == '__main__':    
    from .flexi_dual_grid import FlexiDualGridDataset
    from .sparse_voxel_pbr import SparseVoxelPbrDataset
    
    from .sparse_structure_latent import SparseStructureLatent, ImageConditionedSparseStructureLatent
    from .structured_latent import SLat, ImageConditionedSLat
    from .structured_latent_shape import SLatShape, ImageConditionedSLatShape, MultiImageConditionedSLatShape
    from .structured_latent_svpbr import SLatPbr, ImageConditionedSLatPbr
    

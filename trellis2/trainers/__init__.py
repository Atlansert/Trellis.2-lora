"""trainers 包入口：按需（惰性）导出各训练器类与子模块。

职责：在 __attributes 中登记「类名 -> 相对子模块路径」，首次访问属性时才 import，
避免一次性加载全部训练器（含 VAE / flow matching / transformers 等重依赖）。
输入：`from trellis2.trainers import XXXTrainer` 之类的属性访问。
输出：对应的训练器类，或子模块本身。
契约：新增训练器需同时登记到 __attributes；类名必须与目标模块内定义的名字一致。
"""

import importlib

# 惰性导入表：键为对外暴露的名字，值为相对本包的子模块路径
__attributes = {
    'BasicTrainer': 'basic',
    
    'SparseStructureVaeTrainer': 'vae.sparse_structure_vae',
    'ShapeVaeTrainer': 'vae.shape_vae',
    'PbrVaeTrainer': 'vae.pbr_vae',
    
    'FlowMatchingTrainer': 'flow_matching.flow_matching',
    'FlowMatchingCFGTrainer': 'flow_matching.flow_matching',
    'TextConditionedFlowMatchingCFGTrainer': 'flow_matching.flow_matching',
    'ImageConditionedFlowMatchingCFGTrainer': 'flow_matching.flow_matching',
    
    'SparseFlowMatchingTrainer': 'flow_matching.sparse_flow_matching',
    'SparseFlowMatchingCFGTrainer': 'flow_matching.sparse_flow_matching',
    'TextConditionedSparseFlowMatchingCFGTrainer': 'flow_matching.sparse_flow_matching',
    'ImageConditionedSparseFlowMatchingCFGTrainer': 'flow_matching.sparse_flow_matching',
    'MultiImageConditionedSparseFlowMatchingCFGTrainer': 'flow_matching.sparse_flow_matching',
    
    'DinoV2FeatureExtractor': 'flow_matching.mixins.image_conditioned',
    'DinoV3FeatureExtractor': 'flow_matching.mixins.image_conditioned',
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


# For Pylance（仅供静态检查/直接运行本文件时补全符号，不参与运行时导入路径）
if __name__ == '__main__':
    from .basic import BasicTrainer

    from .vae.sparse_structure_vae import SparseStructureVaeTrainer
    from .vae.shape_vae import ShapeVaeTrainer
    from .vae.pbr_vae import PbrVaeTrainer
    
    from .flow_matching.flow_matching import (
        FlowMatchingTrainer,
        FlowMatchingCFGTrainer,
        TextConditionedFlowMatchingCFGTrainer,
        ImageConditionedFlowMatchingCFGTrainer,
    )
    
    from .flow_matching.sparse_flow_matching import (
        SparseFlowMatchingTrainer,
        SparseFlowMatchingCFGTrainer,
        TextConditionedSparseFlowMatchingCFGTrainer,
        ImageConditionedSparseFlowMatchingCFGTrainer,
        MultiImageConditionedSparseFlowMatchingCFGTrainer,
    )
    
    from .flow_matching.mixins.image_conditioned import (
        DinoV2FeatureExtractor,
        DinoV3FeatureExtractor,
    )

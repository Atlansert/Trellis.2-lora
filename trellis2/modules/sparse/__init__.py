"""稀疏张量子包入口：按属性名懒加载各子模块（避免循环导入与一次性加载开销）。

__attributes 建立“对外符号 -> 所在子模块文件”的映射，首次访问时由 __getattr__
importlib 导入并缓存到 globals()。核心概念：VarLenTensor（变长批量稀疏张量）、
SparseTensor（带 coords 的稀疏张量，coords 每行 4 列 (batch, x, y, z)）。
"""

from . import config
import importlib

__attributes = {
    'VarLenTensor': 'basic',
    'varlen_cat': 'basic',
    'varlen_unbind': 'basic',
    'SparseTensor': 'basic',
    'sparse_cat': 'basic',
    'sparse_unbind': 'basic',
    'SparseGroupNorm': 'norm',
    'SparseLayerNorm': 'norm',
    'SparseGroupNorm32': 'norm',
    'SparseLayerNorm32': 'norm',
    'SparseReLU': 'nonlinearity',
    'SparseSiLU': 'nonlinearity',
    'SparseGELU': 'nonlinearity',
    'SparseActivation': 'nonlinearity',
    'SparseLinear': 'linear',
    'sparse_scaled_dot_product_attention': 'attention',
    'SerializeMode': 'attention',
    'sparse_serialized_scaled_dot_product_self_attention': 'attention',
    'sparse_windowed_scaled_dot_product_self_attention': 'attention',
    'sparse_windowed_scaled_dot_product_cross_attention': 'attention',
    'SparseRotaryPositionEmbedder':  'attention',
    'SparseMultiHeadAttention': 'attention',
    'SparseConv3d': 'conv',
    'SparseInverseConv3d': 'conv',
    'SparseDownsample': 'spatial',
    'SparseUpsample': 'spatial',
    'SparseSubdivide': 'spatial',
    'SparseSpatial2Channel': 'spatial',
    'SparseChannel2Spatial': 'spatial',
    'sparse_nearest_interpolate': 'spatial',
    'sparse_trilinear_interpolate': 'spatial',
    'encode_seq': 'serialize',
    'decode_seq': 'serialize',
}

__submodules = ['transformer', 'conv']

__all__ = list(__attributes.keys()) + __submodules

def __getattr__(name):
    # 惰性导入：首次访问某符号时才 import 对应子模块并把属性挂到本包上
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
    from .basic import *
    from .norm import *
    from .nonlinearity import *
    from .linear import *
    from .attention import *
    from .conv import *
    from .spatial import *
    from .serialize import *
    import transformer
    import conv

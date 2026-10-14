"""Trellis2 推理 pipeline 包的入口。

按名字延迟导入具体 pipeline 类，避免 import 本包时就把全部重依赖拉起来；
并提供统一的 from_pretrained(path) 工厂（按 pipeline.json 的 name 字段选择类）。
"""
import importlib

# 类名 -> 所在模块名，供 __getattr__ 做延迟导入
__attributes = {
    "Trellis2ImageTo3DPipeline": "trellis2_image_to_3d",
    "Trellis2TexturingPipeline": "trellis2_texturing",
}

# 需要延迟导入的子包
__submodules = ['samplers', 'rembg']

__all__ = list(__attributes.keys()) + __submodules

def __getattr__(name):
    # 首次访问该属性时才真正 import，并写入 globals() 缓存（模块级惰性加载）
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


def from_pretrained(path: str):
    """
    Load a pipeline from a model folder or a Hugging Face model hub.

    先定位 pipeline.json（本地优先，否则从 HF 下载），
    再按其中的 name 字段取到对应类并委托给它的 from_pretrained。

    Args:
        path: The path to the model. Can be either local path or a Hugging Face model name.
    """
    import os
    import json
    is_local = os.path.exists(f"{path}/pipeline.json")

    if is_local:
        config_file = f"{path}/pipeline.json"
    else:
        from huggingface_hub import hf_hub_download
        config_file = hf_hub_download(path, "pipeline.json")

    with open(config_file, 'r') as f:
        config = json.load(f)
    return globals()[config['name']].from_pretrained(path)


# For PyLance：仅为静态检查补出显式导入（正常运行时不走这里）
if __name__ == '__main__':
    from . import samplers, rembg
    from .trellis2_image_to_3d import Trellis2ImageTo3DPipeline
    from .trellis2_texturing import Trellis2TexturingPipeline

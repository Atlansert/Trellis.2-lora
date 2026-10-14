"""TRELLIS.2 模型包入口：按名字懒加载各子模块中的模型类。

关键约定：
- __attributes 把对外暴露的类名映射到所属子模块，import 时才真正加载（避免全量依赖）。
- from_pretrained 读取 <path>.json(结构) + <path>.safetensors(权重)，缺一即报错，不做静默回退。
- 检查点严格校验：缺失/多余 key 直接抛 CheckpointValidationError；只有 rope_phases
  这类由配置推出的 buffer 允许重新计算，不被旧权重覆盖。
"""
import importlib

# 类名 -> 子模块名 的映射表，配合下方 __getattr__ 实现懒加载
__attributes = {
    # Sparse Structure
    'SparseStructureEncoder': 'sparse_structure_vae',
    'SparseStructureDecoder': 'sparse_structure_vae',
    'SparseStructureFlowModel': 'sparse_structure_flow',
    
    # SLat Generation
    'SLatFlowModel': 'structured_latent_flow',
    'ElasticSLatFlowModel': 'structured_latent_flow',
    
    # SC-VAEs
    'SparseUnetVaeEncoder': 'sc_vaes.sparse_unet_vae',
    'SparseUnetVaeDecoder': 'sc_vaes.sparse_unet_vae',
    'FlexiDualGridVaeEncoder': 'sc_vaes.fdg_vae',
    'FlexiDualGridVaeDecoder': 'sc_vaes.fdg_vae'
}

__submodules = []

__all__ = list(__attributes.keys()) + __submodules

def __getattr__(name):
    # 首次访问某类名时才 import 对应子模块，并缓存进 globals()
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


class CheckpointValidationError(RuntimeError):
    """Checkpoint does not match the configured architecture; never fall back."""


def from_pretrained(path: str, **kwargs):
    """从预训练检查点构建模型（本地或 HuggingFace 仓库路径）。
    Load a model from a pretrained checkpoint.

    Args:
        path: The path to the checkpoint. Can be either local path or a Hugging Face model name.
              NOTE: config file and model file should take the name f'{path}.json' and f'{path}.safetensors' respectively.
        **kwargs: Additional arguments for the model constructor.
    """
    import os
    import json
    from safetensors.torch import load_file
    is_local = os.path.exists(f"{path}.json") and os.path.exists(f"{path}.safetensors")
    has_local_part = os.path.exists(f"{path}.json") or os.path.exists(f"{path}.safetensors")
    if not is_local and (has_local_part or os.path.isabs(path) or path.startswith(("./", "../"))):
        raise FileNotFoundError(f"Incomplete local checkpoint: {path}.json + .safetensors required")

    if is_local:
        config_file = f"{path}.json"
        model_file = f"{path}.safetensors"
    else:
        from huggingface_hub import hf_hub_download
        path_parts = path.split('/')
        if len(path_parts)<3 or not all(path_parts[:2]):
            raise FileNotFoundError(f'Checkpoint path not local or explicit repo/model: {path}')
        repo_id = f'{path_parts[0]}/{path_parts[1]}'
        model_name = '/'.join(path_parts[2:])
        config_file = hf_hub_download(repo_id, f"{model_name}.json")
        model_file = hf_hub_download(repo_id, f"{model_name}.safetensors")

    with open(config_file, 'r') as f:
        config = json.load(f)
    model = __getattr__(config['name'])(**config['args'], **kwargs)
    state = load_file(model_file)
    # Current computed complex RoPE is regenerated from the configured model;
    # do not let an old serialized/real-cast buffer overwrite it.
    # rope_phases 是按配置重新生成的复数 RoPE 相位缓存，丢弃权重里的旧版本
    buffers = dict(model.named_buffers())
    allowed = {name for name in buffers if name == 'rope_phases'}
    for name in allowed:
        state.pop(name, None)
    result = model.load_state_dict(state, strict=False)
    bad_missing = set(result.missing_keys) - allowed
    if bad_missing or result.unexpected_keys:
        raise CheckpointValidationError(
            f"Checkpoint {model_file} mismatch: missing={sorted(bad_missing)}, "
            f"unexpected={result.unexpected_keys}")
    model._checkpoint_source = {'config_file': os.path.abspath(config_file),
                                'model_file': os.path.abspath(model_file),
                                'requested': path,
                                'computed_buffers_regenerated': sorted(allowed)}

    return model


# For Pylance
if __name__ == '__main__':
    from .sparse_structure_vae import SparseStructureEncoder, SparseStructureDecoder
    from .sparse_structure_flow import SparseStructureFlowModel
    from .structured_latent_flow import SLatFlowModel, ElasticSLatFlowModel
        
    from .sc_vaes.sparse_unet_vae import SparseUnetVaeEncoder, SparseUnetVaeDecoder
    from .sc_vaes.fdg_vae import FlexiDualGridVaeEncoder, FlexiDualGridVaeDecoder

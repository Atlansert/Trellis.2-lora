"""Pipeline 基类。

职责：持有各子模型（统一置 eval）、按 pipeline.json 解析并加载组件权重
（本地目录或 Hugging Face 仓库）、提供统一的设备迁移接口。

权重解析刻意保持严格：组件缺失或加载失败直接报错，不用宽泛异常
回退到其它权重，避免把架构/权重问题悄悄掩盖掉。
"""
from typing import *
import torch
import torch.nn as nn
from .. import models


class Pipeline:
    """
    A base class for pipelines.

    只负责“组件 + 设备”，具体推理流程由子类实现（如 run）。
    """
    def __init__(
        self,
        models: dict[str, nn.Module] = None,
    ):
        # models 为 None 时直接返回，便于构造“空壳”对象后再由 from_pretrained 填充
        if models is None:
            return
        self.models = models
        # 推理用不到梯度，统一切到 eval
        for model in self.models.values():
            model.eval()

    @classmethod
    def from_pretrained(cls, path: str, config_file: str = "pipeline.json") -> "Pipeline":
        """
        Load a pretrained model.

        从 path（本地目录或 HF 仓库）读 pipeline.json，按其中的 models 字段
        逐个解析组件路径并加载。
        """
        import os
        import json
        # 本地目录里有该配置就用本地，否则把 path 当作 HF 仓库名去下载
        is_local = os.path.exists(f"{path}/{config_file}")

        if is_local:
            config_file = f"{path}/{config_file}"
        else:
            from huggingface_hub import hf_hub_download
            config_file = hf_hub_download(path, config_file)

        with open(config_file, 'r') as f:
            args = json.load(f)['args']

        _models = {}
        for k, v in args['models'].items():
            # 只加载子类声明需要的组件（子类用 model_names_to_load 过滤）
            if hasattr(cls, 'model_names_to_load') and k not in cls.model_names_to_load:
                continue
            candidate = os.path.join(path, v)
            local_part = os.path.exists(f"{candidate}.json") or os.path.exists(f"{candidate}.safetensors")
            if local_part:
                component_path = candidate
            elif os.path.isabs(v) or v.startswith(("ckpts/", "./", "../")):
                # 配置里写的是绝对路径，或相对项目根的 ckpts/ ./ ../ 路径
                if is_local:
                    raise FileNotFoundError(f"Pipeline component {k} missing locally: {candidate}")
                component_path = f"{path}/{v}"
            elif len(v.split('/')) >= 3:
                # Explicit cross-repository component such as the SS decoder.
                # 形如 org/repo/子目录 的显式跨仓库组件
                component_path = v
            elif is_local:
                raise FileNotFoundError(f"Pipeline component {k} missing locally: {candidate}")
            else:
                component_path = f"{path}/{v}"
            # Architecture/weight errors must propagate; never mask a broken
            # local component with broad exception fallback to other weights.
            # 中文：架构/权重错误必须向上抛，禁止用宽泛异常回退到别的权重。
            _models[k] = models.from_pretrained(component_path)

        new_pipeline = cls(_models)
        new_pipeline._pretrained_args = args
        return new_pipeline

    @property
    def device(self) -> torch.device:
        # 优先用子模型自报的 device，退而取首个参数的 device
        if hasattr(self, '_device'):
            return self._device
        for model in self.models.values():
            if hasattr(model, 'device'):
                return model.device
        for model in self.models.values():
            if hasattr(model, 'parameters'):
                return next(model.parameters()).device
        raise RuntimeError("No device found.")

    def to(self, device: torch.device) -> None:
        # 整体迁移：把所有子模型搬到指定设备
        for model in self.models.values():
            model.to(device)

    def cuda(self) -> None:
        self.to(torch.device("cuda"))

    def cpu(self) -> None:
        self.to(torch.device("cpu"))
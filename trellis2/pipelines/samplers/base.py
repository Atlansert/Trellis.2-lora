"""采样器抽象基类。"""
from typing import *
from abc import ABC, abstractmethod


class Sampler(ABC):
    """
    A base class for samplers.

    约定接口：入参为待采样的模型与初始噪声，返回含 samples 的结果对象。
    """

    @abstractmethod
    def sample(
        self,
        model,
        **kwargs
    ):
        """
        Sample from a model.

        抽象接口，具体实现见 flow_euler.py 中的各 Euler 采样器。
        """
        pass
    
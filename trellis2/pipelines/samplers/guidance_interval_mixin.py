"""Guidance interval mixin。

只在时间步 t 落入 guidance_interval 区间时才启用 CFG，区间外退化为
纯条件预测（等价 guidance_strength=1），可减少高噪声段被 CFG 放大出的伪影。
"""
from typing import *


class GuidanceIntervalSamplerMixin:
    """
    A mixin class for samplers that apply classifier-free guidance with interval.

    需与 ClassifierFreeGuidanceSamplerMixin 叠加使用（它负责真正的 CFG 计算）。
    """

    def _inference_model(self, model, x_t, t, cond, guidance_strength, guidance_interval, **kwargs):
        if guidance_interval[0] <= t <= guidance_interval[1]:
            # 区间内：按给定强度做 CFG
            return super()._inference_model(model, x_t, t, cond, guidance_strength=guidance_strength, **kwargs)
        else:
            # 区间外：强度置 1，即跳过 CFG
            return super()._inference_model(model, x_t, t, cond, guidance_strength=1, **kwargs)

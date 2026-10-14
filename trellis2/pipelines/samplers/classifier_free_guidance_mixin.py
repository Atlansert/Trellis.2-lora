"""Classifier-free guidance（CFG）mixin。

对同一批噪声分别用正/负条件各跑一次模型，再按 guidance_strength 线性插值；
可选 CFG rescale，把插值结果的尺度拉回正条件预测的尺度以抑制过曝/过饱和。
"""
from typing import *


class ClassifierFreeGuidanceSamplerMixin:
    """
    A mixin class for samplers that apply classifier-free guidance.

    需与实现 _inference_model/_pred_to_xstart/_xstart_to_pred 的采样器配合使用。
    """

    def _inference_model(self, model, x_t, t, cond, neg_cond, guidance_strength, guidance_rescale=0.0, **kwargs):
        if guidance_strength == 1:
            # 强度 1 = 等价于关闭 CFG，只跑一次正条件前向
            return super()._inference_model(model, x_t, t, cond, **kwargs)
        elif guidance_strength == 0:
            # 强度 0 = 完全走负条件（无条件）预测
            return super()._inference_model(model, x_t, t, neg_cond, **kwargs)
        else:
            # 常规情形：正负条件各一次前向，再加权融合
            pred_pos = super()._inference_model(model, x_t, t, cond, **kwargs)
            pred_neg = super()._inference_model(model, x_t, t, neg_cond, **kwargs)
            pred = guidance_strength * pred_pos + (1 - guidance_strength) * pred_neg
            
            # CFG rescale
            if guidance_rescale > 0:
                # 把 CFG 结果的逐样本 std 缩放到正条件预测的 std，再按比例混合回去
                x_0_pos = self._pred_to_xstart(x_t, t, pred_pos)
                x_0_cfg = self._pred_to_xstart(x_t, t, pred)
                std_pos = x_0_pos.std(dim=list(range(1, x_0_pos.ndim)), keepdim=True)
                std_cfg = x_0_cfg.std(dim=list(range(1, x_0_cfg.ndim)), keepdim=True)
                x_0_rescaled = x_0_cfg * (std_pos / std_cfg)
                x_0 = guidance_rescale * x_0_rescaled + (1 - guidance_rescale) * x_0_cfg
                pred = self._xstart_to_pred(x_t, t, x_0)
                
            return pred

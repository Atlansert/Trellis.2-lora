"""Classifier-Free Guidance（CFG）训练 mixin。

职责：训练时按概率 p_uncond 把条件替换成负条件（空条件），使模型同时学会有/无条件预测；
推理时同时给出 cond 与 neg_cond，由采样器做引导。
输入：cond（Tensor / list / dict）与 neg_cond；输出：可能已被部分替换的 cond，
或在推理路径下打包成 {'cond':..., 'neg_cond':...} 供采样器使用。
契约：使用本 mixin 必须提供 neg_cond；p_uncond=0 时不做任何丢弃，等价于普通条件训练。
"""

import torch
import numpy as np
from ....utils.general_utils import dict_foreach
from ....pipelines import samplers


class ClassifierFreeGuidanceMixin:
    """按 p_uncond 随机丢弃条件（换成 neg_cond），并给出支持引导区间/重缩放的采样器。"""

    def __init__(self, *args, p_uncond: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.p_uncond = p_uncond   # 条件被替换为负条件的概率（CFG 训练强度）

    def get_cond(self, cond, neg_cond=None, **kwargs):
        """
        Get the conditioning data.
        训练用：按 p_uncond 逐样本随机把 cond 换成 neg_cond。
        cond 支持 Tensor / list / dict（dict 会递归按同样掩码替换）。
        """
        assert neg_cond is not None, "neg_cond must be provided for classifier-free guidance" 

        if self.p_uncond > 0:
            # randomly drop the class label
            # 取 batch 大小：只用第一个条件张量/列表判断 B（用于生成丢弃掩码）
            def get_batch_size(cond):
                if isinstance(cond, torch.Tensor):
                    return cond.shape[0]
                elif isinstance(cond, list):
                    return len(cond)
                else:
                    raise ValueError(f"Unsupported type of cond: {type(cond)}")
                
            ref_cond = cond if not isinstance(cond, dict) else cond[list(cond.keys())[0]]
            B = get_batch_size(ref_cond)
            
            # 按掩码在 cond / neg_cond 之间逐样本选择；Tensor 需把掩码扩到与 cond 同秩
            def select(cond, neg_cond, mask):
                if isinstance(cond, torch.Tensor):
                    mask = torch.tensor(mask, device=cond.device).reshape(-1, *[1] * (cond.ndim - 1))
                    return torch.where(mask, neg_cond, cond)
                elif isinstance(cond, list):
                    return [nc if m else c for c, nc, m in zip(cond, neg_cond, mask)]
                else:
                    raise ValueError(f"Unsupported type of cond: {type(cond)}")
            
            # mask[i]=True 表示第 i 个样本丢弃条件
            mask = list(np.random.rand(B) < self.p_uncond)
            if not isinstance(cond, dict):
                cond = select(cond, neg_cond, mask)
            else:
                cond = dict_foreach([cond, neg_cond], lambda x: select(x[0], x[1], mask))
    
        return cond

    def get_inference_cond(self, cond, neg_cond=None, **kwargs):
        """
        Get the conditioning data for inference.
        推理用：不做丢弃，直接把 cond 与 neg_cond 一起交给采样器。
        """
        assert neg_cond is not None, "neg_cond must be provided for classifier-free guidance"
        return {'cond': cond, 'neg_cond': neg_cond, **kwargs}
    
    def get_sampler(self, **kwargs) -> samplers.FlowEulerGuidanceIntervalSampler:
        """
        Get the sampler for the diffusion process.

        使用与线上推理一致的 FlowEulerGuidanceIntervalSampler：只有它支持
        guidance_interval / guidance_rescale。默认参数（interval=(0,1)、
        rescale=0）下行为与旧的 FlowEulerCfgSampler 等价，因此未显式配置
        snapshot_sampler_params 的历史训练不受影响。
        """
        return samplers.FlowEulerGuidanceIntervalSampler(self.sigma_min)

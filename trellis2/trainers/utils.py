"""训练公共工具：dtype 名称解析、FP32 主参数（master params）辅助函数、LR 预热调度器。

职责：混合精度训练时维护一份 FP32 主参数副本（更新用 FP32，前向用半精度），
并提供梯度/参数在主参数与模型参数之间的搬运，以及线性 warmup 学习率调度。
输入/输出：均直接操作传入的 Parameter 列表，不返回新模型。
契约：master_params 是 _flatten_dense_tensors 得到的单个扁平 Parameter（长度=所有参数元素数之和），
调用方必须保证 model_params 的顺序在 flatten/unflatten 之间保持一致。
"""

import torch
import torch.nn as nn

# FP16 utils
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors


def str_to_dtype(dtype_str: str):
    # 把配置里的字符串（f16/fp16/bf16/...）映射为 torch.dtype，未知名称直接 KeyError
    return {
        'f16': torch.float16,
        'fp16': torch.float16,
        'float16': torch.float16,
        'bf16': torch.bfloat16,
        'bfloat16': torch.bfloat16,
        'f32': torch.float32,
        'fp32': torch.float32,
        'float32': torch.float32,
    }[dtype_str]
    

def make_master_params(model_params):
    """
    Copy model parameters into a inflated tensor of full-precision parameters.
    把模型参数（可能是半精度）复制、展平并拼成一个 FP32 主参数副本，用于稳定更新。
    """
    master_params = _flatten_dense_tensors(
        [param.detach().float() for param in model_params]
    )
    master_params = nn.Parameter(master_params)
    master_params.requires_grad = True
    return [master_params]


def unflatten_master_params(model_params, master_params):
    """
    Unflatten the master parameters to look like model_params.
    按 model_params 的形状把主参数切片还原成参数列表（只读 detach 视图）。
    """
    return _unflatten_dense_tensors(master_params[0].detach(), model_params)


def model_params_to_master_params(model_params, master_params):
    """
    Copy the model parameter data into the master parameters.
    把当前模型参数（转 FP32 后）写回主参数，通常在每步更新前调用。
    """
    master_params[0].detach().copy_(
        _flatten_dense_tensors([param.detach().float() for param in model_params])
    )


def master_params_to_model_params(model_params, master_params):
    """
    Copy the master parameter data back into the model parameters.
    优化器更新完主参数后，把 FP32 主参数按原 dtype 拷回模型参数。
    """
    for param, master_param in zip(
        model_params, _unflatten_dense_tensors(master_params[0].detach(), model_params)
    ):
        param.detach().copy_(master_param)


def model_grads_to_master_grads(model_params, master_params):
    """
    Copy the gradients from the model parameters into the master parameters
    from make_master_params().
    把各模型参数的梯度转 FP32 后展平，挂到主参数上（供优化器更新主参数）。
    """
    master_params[0].grad = _flatten_dense_tensors(
        [param.grad.data.detach().float() for param in model_params]
    )
    

def zero_grad(model_params):
    # 就地清零梯度：先 detach 断开计算图（或关闭 requires_grad），再 zero_()
    for param in model_params:
       if param.grad is not None:
            if param.grad.grad_fn is not None:
                param.grad.detach_()
            else:
                param.grad.requires_grad_(False)
            param.grad.zero_()
            

# LR Schedulers
from torch.optim.lr_scheduler import LambdaLR

class LinearWarmupLRScheduler(LambdaLR):
    """线性 warmup 调度器：前 warmup_steps 步 LR 从 1/warmup 线性升到基准值，之后恒为 1 倍。"""

    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        super(LinearWarmupLRScheduler, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)
        
    def lr_lambda(self, current_step):
        # 返回的是相对基准 LR 的倍率；注意用 current_step+1 保证首步不为 0
        if current_step < self.warmup_steps:
            return float(current_step + 1) / self.warmup_steps
        return 1.0
        
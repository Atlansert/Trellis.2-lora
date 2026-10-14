"""注意力后端全局配置（dense 版）。

BACKEND 决定 scaled_dot_product_attention 走哪个实现：xformers / flash_attn /
flash_attn_3 / sdpa（PyTorch 原生）/ naive（纯手写，仅调试用）。
导入本模块时即从环境变量读取一次：ATTN_BACKEND、ATTN_DEBUG=1。
"""

from typing import *

BACKEND = 'flash_attn'  # 默认后端
DEBUG = False

def __from_env():
    import os
    
    global BACKEND
    global DEBUG
    
    env_attn_backend = os.environ.get('ATTN_BACKEND')
    env_attn_debug = os.environ.get('ATTN_DEBUG')
    
    if env_attn_backend is not None and env_attn_backend in ['xformers', 'flash_attn', 'flash_attn_3', 'sdpa', 'naive']:
        BACKEND = env_attn_backend
    if env_attn_debug is not None:
        DEBUG = env_attn_debug == '1'

    print(f"[ATTENTION] Using backend: {BACKEND}")
        

__from_env()
    

def set_backend(backend: Literal['xformers', 'flash_attn']):
    global BACKEND
    BACKEND = backend

def set_debug(debug: bool):
    global DEBUG
    DEBUG = debug

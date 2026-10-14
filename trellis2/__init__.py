"""TRELLIS.2 顶层包。

统一导出各子包：
- models：网络结构（flow 模型、各类 decoder/encoder）
- modules：基础算子与模块（稀疏张量、LoRA、图像特征提取等）
- pipelines：推理主干（图 → 3D、网格贴图）
- renderers / representations：渲染与几何/材质表示
- utils：训练与渲染等公共工具
"""
from . import models
from . import modules
from . import pipelines
from . import renderers
from . import representations
from . import utils

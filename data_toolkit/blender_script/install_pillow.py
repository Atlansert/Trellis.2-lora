"""在 Blender 内置 Python 里安装 Pillow（无 CLI 参数）。

- 用途：Blender 自带的 Python 通常没有 pip/Pillow，而 dump_pbr.py 导出贴图需要 Pillow；
  本脚本在 Blender 解释器内引导 pip 并安装依赖（需要联网）。
- 调用方式：由 data_toolkit/dump_pbr.py 的 _install_blender() 执行
  `blender -b --python blender_script/install_pillow.py`。
- 副作用：向 Blender 的 site-packages 安装 Pillow；重复运行只是重复安装（幂等性由 pip 保证）。
"""
import subprocess
import sys
import ensurepip

# 先用 ensurepip 引导出 pip，再用当前解释器安装 Pillow（只作用于 Blender 内置 Python）。
ensurepip.bootstrap()
subprocess.check_call([sys.executable, "-m", "pip", "install", "Pillow"])
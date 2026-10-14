"""Explicit right-handed common frame for NEW data only.

Native glTF Y-up -> Blender/pipeline raw Z-up, then view yaw.
Raw length slot is y; formal GLB conversion (x,z,-y) makes length z/up y.
OBJ/unknown-up inputs must be separately calibrated; no implicit fallback.
This does NOT rotate existing generated meshes or transpose encoded latents.

中文说明：
- 用途：为“新数据”定义唯一显式的右手公共帧（规范帧），供数据重建与 latent 编码共用。
- 输入/输出：Nx3 顶点或整数体素坐标（+ yaw 角度）→ 变换后的 Nx3 数组；非法输入直接抛 ValueError。
- 关键约定：native glTF Y-up → 内部 (x,-z,y)（行列式 +1 的正确旋转），再绕 yaw 旋转；raw 轴 = (depth, length, up)。
- 正式 GLB 用 (x,z,-y)；OBJ/未知 up 的输入必须另行标定，不做隐式兜底。
- 只做坐标变换：不旋转已生成的网格，也不转置已编码的 latent。
- 无 CLI，作为库被其它工具/测试导入（见 tests/test_canonical_frame.py）。
"""
import numpy as np
FRAME_VERSION='trellis.common-frame.gltf-y-up.v1'

# 规范帧主变换：native glTF Y-up 顶点 → 内部 (x,-z,y)，再绕 z 轴按 yaw 旋转。
# vertices: Nx3 有限顶点；yaw_degrees: 视角偏航角（度）；返回 Nx3 float64。
def native_gltf_to_internal(vertices,yaw_degrees=0.0):
    v=np.asarray(vertices)
    if v.ndim!=2 or v.shape[1]!=3 or not np.isfinite(v).all():
        raise ValueError('finite native glTF Nx3 vertices required')
    if not np.isfinite(yaw_degrees):raise ValueError('finite yaw required')
    # 轴交换 (x,z,y) 再把 y 取负：得到 (x,-z,y)，行列式为 +1 的真旋转。
    base=v[:,[0,2,1]].astype(np.float64)*np.array([1.,-1.,1.])
    a=np.deg2rad(yaw_degrees);c,s=np.cos(a),np.sin(a)
    rz=np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
    return base@rz

# 旧版 SS 体素支撑集坐标 → 内部坐标系：仅用于诊断/重新编码的输入，绝不用于变换 SS 均值。
def legacy_ss_support_to_internal(coords,resolution=64):
    """Diagnostic/re-encoding INPUT support only; never transform SS means.
    Existing SS index0=down,index1=length,index2=negative depth.

    中文说明：把旧约定（index0=下、index1=长度、index2=负深度）的整数体素坐标映射到
    内部帧；输出在网格上是双射（对合），不是特征变换。resolution 为网格边长。
    """
    c=np.asarray(coords)
    if resolution<=1 or c.ndim!=2 or c.shape[1]!=3 or not np.isfinite(c).all():
        raise ValueError('valid Nx3 support and grid resolution required')
    if not np.equal(c,np.floor(c)).all() or (c<0).any() or (c>=resolution).any():
        raise ValueError('integer grid support required')
    out=c[:,[2,1,0]].copy()
    # 前两轴取补（resolution-1-x），保证再次调用可还原成原坐标。
    out[:,[0,2]]=resolution-1-out[:,[0,2]]
    return out

# 内部帧顶点 → 正式 GLB 约定 (x,z,-y)；与 native_gltf_to_internal 在 yaw=0 时互逆。
def internal_to_formal_glb(vertices):
    v=np.asarray(vertices)
    if v.ndim!=2 or v.shape[1]!=3 or not np.isfinite(v).all():raise ValueError('finite Nx3 vertices required')
    return v[:,[0,2,1]]*np.array([1.,1.,-1.])
